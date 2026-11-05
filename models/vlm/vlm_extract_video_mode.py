#!/usr/bin/env python3
import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

import transformers
from transformers import AutoProcessor

try:
    from qwen_vl_utils import process_vision_info
except ImportError as exc:
    raise SystemExit("qwen_vl_utils is required for video mode: pip install qwen-vl-utils") from exc


EMOTION_LABELS = ["neutral", "happy", "fearful"]
DEFAULT_PROMPT = (
    "What is the facial emotion of the person in this video? "
    f"Choose exactly one from: {', '.join(EMOTION_LABELS)}. "
    "Reply with only the label, nothing else."
)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def extract_emotion(text: str) -> str:
    t = (text or "").strip().lower()
    for lab in EMOTION_LABELS:
        if lab in t:
            return lab
    if "fear" in t:
        return "fearful"
    if "happ" in t or "joy" in t:
        return "happy"
    return "unknown"


def load_model(model_id: str, dtype: str, random_weights: bool = False):
    if not torch.cuda.is_available():
        raise SystemExit("[FATAL] no GPU: refusing to run on CPU. Resubmit on another node.")
    torch_dtype = {"float16": torch.float16, "float32": torch.float32}.get(dtype, torch.bfloat16)
    kwargs: Dict[str, Any] = {"torch_dtype": torch_dtype, "device_map": "auto"}
    errors = []
    # Qwen3 first: the 2.5 class can mis-map Qwen3-VL weights
    for cls_name in ("Qwen3VLForConditionalGeneration",
                     "Qwen2_5_VLForConditionalGeneration",
                     "AutoModelForImageTextToText"):
        cls = getattr(transformers, cls_name, None)
        if cls is None:
            continue
        try:
            if random_weights:
                cfg = transformers.AutoConfig.from_pretrained(model_id)
                try:
                    model = cls.from_config(cfg)
                except (AttributeError, TypeError):
                    model = cls(cfg)
                model = model.to(dtype=torch_dtype).to("cuda")
            else:
                model = cls.from_pretrained(model_id, **kwargs)
            model.eval()
            return model, cls_name
        except Exception as exc:
            errors.append(f"{cls_name}: {exc}")
    raise RuntimeError("Could not load a video-capable Qwen class.\n" + "\n".join(errors))


class QwenVideoExtractor:
    def __init__(self, model_id: str, dtype: str = "bfloat16", random_weights: bool = False):
        self.model, self.model_class = load_model(model_id, dtype, random_weights)
        self.processor = AutoProcessor.from_pretrained(model_id)
        print(f"[model] {model_id} loaded as {self.model_class}"
              f"{'  [RANDOM WEIGHTS]' if random_weights else '  [pretrained]'}", flush=True)
        # vision blocks are not in the decoder's hidden_states: hooked, analysis pass only
        self._vision_out: List[torch.Tensor] = []
        self._capture = False
        self.vision_blocks = self._find_vision_blocks()
        for blk in self.vision_blocks:
            blk.register_forward_hook(self._vision_hook)

    def _find_vision_blocks(self):
        for name, module in self.model.named_modules():
            if name.endswith("visual.blocks"):
                return module
        raise RuntimeError("vision encoder blocks (*.visual.blocks) not found")

    def _vision_hook(self, _module, _inputs, output):
        if self._capture:
            self._vision_out.append(output[0] if isinstance(output, tuple) else output)

    def _video_token_id(self) -> int:
        tid = getattr(self.model.config, "video_token_id", None)
        if tid is not None:
            return int(tid)
        for tok in ("<|video_pad|>", "<|vision_pad|>"):
            cand = self.processor.tokenizer.convert_tokens_to_ids(tok)
            if cand is not None and cand >= 0:
                return int(cand)
        raise RuntimeError("Could not determine the video token id.")

    def _merge_size(self) -> int:
        ip = getattr(self.processor, "image_processor", None)
        return int(getattr(ip, "merge_size", 2) or 2)

    @staticmethod
    def canonical_frame_paths(video_path: str, tmpdir: str, indices: List[int]) -> List[str]:
        """Save the manifest frames as JPEGs; each path twice (Qwen fuses frame pairs)."""
        import av
        from PIL import Image

        with av.open(video_path) as container:
            frames = [f.to_ndarray(format="rgb24") for f in container.decode(video=0)]
        total = len(frames)
        if total == 0:
            raise RuntimeError(f"Could not read any frame from {video_path}")
        idx = np.asarray(indices)
        if idx.max() >= total:
            raise RuntimeError(f"{video_path}: frame {idx.max()} requested, {total} decoded")
        paths: List[str] = []
        stem = Path(video_path).stem
        for k, i in enumerate(idx):
            p = os.path.join(tmpdir, f"{stem}_f{k:03d}.jpg")
            Image.fromarray(frames[int(i)]).save(p, quality=95)
            paths.extend([p, p])
        return paths

    @torch.inference_mode()
    def infer_video(self, video_path: str, prompt: str, max_new_tokens: int,
                    tmpdir: str, indices: List[int]) -> Dict[str, Any]:
        nframes = len(indices)
        vid = self.canonical_frame_paths(video_path, tmpdir, indices)

        messages = [{
            "role": "user",
            # without nframes, process_vision_info resamples the list (32 -> 4 frames)
            "content": [{"type": "video", "video": vid, "nframes": len(vid)}] + [
                {"type": "text", "text": prompt},
            ],
        }]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        if not video_inputs:
            raise RuntimeError(f"process_vision_info returned no video for {video_path}")

        # do_sample_frames defaults to True and would resample the list (32 -> 4 frames)
        inputs = self.processor(text=[text], images=image_inputs, videos=video_inputs,
                                padding=True, return_tensors="pt", do_sample_frames=False)
        dev = next(self.model.parameters()).device
        inputs = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in inputs.items()}

        self._vision_out.clear()
        self._capture = True
        try:
            out = self.model(**inputs, output_hidden_states=True, return_dict=True)
        finally:
            self._capture = False
        hidden_states = out.hidden_states
        if hidden_states is None:
            raise RuntimeError("Model returned no hidden_states.")

        ids = inputs["input_ids"][0]
        vid_positions = (ids == self._video_token_id()).nonzero(as_tuple=True)[0]
        if vid_positions.numel() == 0:
            raise RuntimeError("No video tokens found in input_ids.")

        grid = inputs["video_grid_thw"][0].tolist()          # [T, H, W] in patches
        n_groups = int(grid[0])
        merge = self._merge_size()
        per_group = int(grid[1] * grid[2] // (merge * merge))
        if per_group <= 0 or n_groups * per_group != int(vid_positions.numel()):
            per_group = int(vid_positions.numel()) // max(n_groups, 1)
            print(f"  [warn] grid {grid} vs {int(vid_positions.numel())} tokens; "
                  f"falling back to {n_groups}x{per_group}", flush=True)
        if n_groups != nframes:
            raise RuntimeError(
                f"expected {nframes} temporal groups from a {len(vid)}-frame list but "
                f"got {n_groups} (grid {grid}). The processor resampled the frames; "
                f"the model-time axis would not match the EEG windows.")
        usable = n_groups * per_group

        per_layer: List[np.ndarray] = []
        for layer_hidden in hidden_states:                    # (1, seq, hidden)
            vis = layer_hidden[0][vid_positions][:usable]     # (usable, hidden)
            vis = vis.reshape(n_groups, per_group, -1).mean(dim=1)   # (T, hidden)
            vis = F.normalize(vis.float(), dim=-1)
            per_layer.append(vis.cpu().numpy())
        acts = np.stack(per_layer, axis=1)                    # (T, n_layers, hidden)

        # vision patch tokens are time-major: the same T-way split
        if len(self._vision_out) != len(self.vision_blocks):
            raise RuntimeError(f"captured {len(self._vision_out)} vision blocks, "
                               f"expected {len(self.vision_blocks)}")
        per_block: List[np.ndarray] = []
        for block_hidden in self._vision_out:                 # (n_patches, vision_hidden)
            if block_hidden.shape[0] % n_groups:
                raise RuntimeError(f"{block_hidden.shape[0]} vision tokens do not split "
                                   f"into {n_groups} temporal groups")
            vis = block_hidden.reshape(n_groups, -1, block_hidden.shape[-1]).mean(dim=1)
            vis = F.normalize(vis.float(), dim=-1)
            per_block.append(vis.cpu().numpy())
        vision_acts = np.stack(per_block, axis=1)             # (T, n_vision_blocks, vision_hidden)
        self._vision_out.clear()

        gen_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        gen_only = gen_ids[:, inputs["input_ids"].shape[1]:]
        pred_text = self.processor.batch_decode(
            gen_only, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0].strip()

        return {"activations": acts, "vision_activations": vision_acts,
                "n_groups": n_groups, "per_group": per_group,
                "grid_thw": grid, "prediction_text": pred_text,
                "pred_label": extract_emotion(pred_text)}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--metadata_csv", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--model_id", default="Qwen/Qwen3-VL-8B-Instruct")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--random", dest="random_weights", action="store_true",
                   help="random weights (untrained control)")
    p.add_argument("--max_new_tokens", type=int, default=8)
    p.add_argument("--prompt", default=DEFAULT_PROMPT)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    set_seed(args.seed)
    out_dir = Path(args.output_dir)
    act_dir = out_dir / "activations"
    act_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.metadata_csv)
    missing = {"video_path", "stimulus_id", "frame_indices"} - set(df.columns)
    if missing:
        raise ValueError(f"metadata_csv missing columns: {missing}")

    tmp = Path(tempfile.mkdtemp(prefix='qwen_vid_frames_'))
    ex = QwenVideoExtractor(args.model_id, args.dtype, random_weights=args.random_weights)

    rows: List[Dict[str, Any]] = []
    for i, row in df.iterrows():
        video_path = str(row["video_path"])
        cond = str(row["stimulus_id"])
        if not os.path.exists(video_path):
            raise SystemExit(f"ABORT: missing video {video_path}")
        indices = [int(k) for k in str(row["frame_indices"]).split()]
        res = ex.infer_video(video_path, args.prompt, args.max_new_tokens, str(tmp), indices)
        acts = res["activations"]
        np.save(act_dir / f"{cond}.npy", acts)
        np.save(act_dir / f"{cond}_vision.npy", res["vision_activations"])
        print(f"[{i+1}/{len(df)}] {cond}: acts={acts.shape} grid={res['grid_thw']} "
              f"({res['n_groups']} timepoints x {res['per_group']} tok/group) "
              f"pred={res['pred_label']}", flush=True)

        for t in range(acts.shape[0]):
            rows.append({"condition_id": cond, "stimulus_id": cond,
                         "video_path": video_path, "frame_index": t,
                         "prediction_text": res["prediction_text"],
                         "pred_label": res["pred_label"],
                         "n_layers": int(acts.shape[1]), "hidden_size": int(acts.shape[2])})

    if not rows:
        raise RuntimeError("No videos processed - check metadata paths.")
    pd.DataFrame(rows).to_csv(out_dir / "frame_predictions.csv", index=False)
    (out_dir / "summary.json").write_text(json.dumps(
        {"mode": "video", "model_id": args.model_id, "nframes": len(indices),
         "random_weights": bool(args.random_weights), "seed": args.seed,
         "n_conditions": len(set(r["condition_id"] for r in rows))}, indent=2))
    print(f"[ok] wrote {act_dir} and frame_predictions.csv")


if __name__ == "__main__":
    main()
