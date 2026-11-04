"""Shared look of the paper's result figures: colour = training regime, marker = model family.

Regime colours follow the precursor paper (face training blue, generic training magenta,
random grey) plus green for emotion fine-tuning; validated as a set (dataviz validator,
light surface, all pairs). Families are told apart by marker shape, so colour never
carries identity alone.
"""
import matplotlib as mpl

INK, INK2, MUTED, RULE = "#0b0b0b", "#52514e", "#8a8984", "#d6d5d0"
PARTICIPANT = "#c9c8c3"
REGIMES = ["Random", "Generic", "Face", "Emotion"]
REGIME_COLORS = {"Random": MUTED, "Generic": "#e87ba4", "Face": "#2a78d6", "Emotion": "#008300"}
REGIME_OF = {"Random": "Random", "ImageNet": "Generic", "Kinetics-400": "Generic", "Pretrained": "Generic",
             "VGGFace2": "Face", "VGG-Face": "Face", "AffectNet FT": "Emotion", "DFEW FT": "Emotion"}
REGIME_LABELS = {"Random": "random", "Generic": "generic", "Face": "face", "Emotion": "emotion"}
MARKERS = {"ResNet18": "o", "ResNet50": "s", "FaceNet": "^", "VGG16": "D", "M3DFEL": "p", "Qwen3-VL": "*"}
MARKER_SIZE = {"*": 7.5, "p": 5.5, "D": 4.2}


def setup():
    mpl.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7.5, "axes.labelsize": 7.5, "axes.titlesize": 7.5, "xtick.labelsize": 7,
        "ytick.labelsize": 7, "legend.fontsize": 7, "axes.linewidth": 0.6, "axes.edgecolor": INK,
        "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK, "text.color": INK,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
        "pdf.fonttype": 42, "savefig.dpi": 300, "axes.titlepad": 3,
    })


def size(marker):
    return MARKER_SIZE.get(marker, 4.5)


def panel_label(ax, letter, title=None, x=-0.22, y=1.06):
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom", ha="left")
    if title:
        ax.text(x + 0.09, y, title, transform=ax.transAxes, fontsize=7.7, va="bottom", ha="left")


def point_format(x):
    """.05 rather than 0.05, as in the paper."""
    s = f"{x:g}"
    return s.replace("0.", ".", 1) if s.startswith(("0.", "-0.")) else s


def stars(p):
    return "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else "n.s."


def blend(color, t):
    """color mixed with white (t = share of the colour kept)."""
    from matplotlib.colors import to_rgb
    return tuple(t * c + (1 - t) for c in to_rgb(color))
