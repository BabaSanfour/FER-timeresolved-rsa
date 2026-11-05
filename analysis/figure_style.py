import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap

INK, INK2, MUTED, RULE = "#0b0b0b", "#52514e", "#8a8984", "#d6d5d0"
# colour = training regime, marker = model family
REGIMES = ["Random", "Generic", "Face", "Emotion"]
REGIME_COLORS = {"Random": MUTED, "Generic": "#e87ba4", "Face": "#2a78d6", "Emotion": "#008300"}
REGIME_OF = {"Random": "Random", "ImageNet": "Generic", "Kinetics-400": "Generic", "Pretrained": "Generic",
             "VGGFace2": "Face", "VGG-Face": "Face", "AffectNet FT": "Emotion", "DFEW FT": "Emotion"}
REGIME_LABELS = {"Random": "random", "Generic": "generic pretraining", "Face": "face recognition",
                 "Emotion": "emotion recognition"}
MARKERS = {"ResNet18": "o", "ResNet50": "s", "FaceNet": "^", "VGG16": "D", "M3DFEL": "p", "Qwen3-VL": "*"}
MARKER_SIZE = {"*": 7.5, "p": 5.5, "D": 4.2}
REGION_COLORS = {"Frontal": "#6250d6", "Central": "#e87ba4", "CentroParietal": "#008300",
                 "OccipitoTemporal": "#eda100", "wholescalp": INK}
DIVERGING = LinearSegmentedColormap.from_list("div", ["#2a78d6", "#f0efec", "#e34948"])


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


def point_format(x):
    """.05 rather than 0.05."""
    s = f"{x:g}"
    return s.replace("0.", ".", 1) if s.startswith(("0.", "-0.")) else s


def point_ticks(ax, axis="y"):
    """Tick labels without the leading zero."""
    getattr(ax, f"{axis}axis").set_major_formatter(lambda v, _: point_format(round(v, 3)))


def black_text(fig):
    """Every text of the figure in black."""
    from matplotlib.text import Text
    for t in fig.findobj(Text):
        t.set_color(INK)


def save(fig, out):
    """All text in black; writes <out>.pdf and <out>.png."""
    black_text(fig)
    for ext in ("pdf", "png"):
        fig.savefig(f"{out}.{ext}")
