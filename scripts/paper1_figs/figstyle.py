"""Self-contained plotting defaults; no user-level helper installation needed."""
from pathlib import Path
import matplotlib as mpl


def apply(width=5.5, aspect=.618, nrows=1, ncols=1, base=9., korean=False):
    widths = {'single': 3.5, 'double': 7.16, 'text': 5.5}
    width = widths[width] if isinstance(width, str) else width
    size = (width, width / ncols * aspect * nrows)
    serif = ['Times New Roman', 'Nimbus Roman', 'Liberation Serif', 'STIXGeneral', 'DejaVu Serif']
    if korean:
        serif = ['NanumMyeongjo', 'HCR Batang', 'Haansoft Batang'] + serif
    mpl.rcParams.update({'font.family': 'serif', 'font.serif': serif,
        'mathtext.fontset': 'stix', 'font.size': base, 'axes.labelsize': base+1,
        'axes.titlesize': base, 'xtick.labelsize': base-1, 'ytick.labelsize': base-1,
        'legend.fontsize': base-1, 'legend.frameon': False, 'axes.linewidth': .6,
        'lines.linewidth': 1.2, 'lines.markersize': 4, 'xtick.major.width': .6,
        'ytick.major.width': .6, 'xtick.direction': 'out', 'ytick.direction': 'out',
        'axes.unicode_minus': not korean, 'figure.figsize': size, 'figure.dpi': 100,
        'figure.constrained_layout.use': True, 'savefig.bbox': 'tight',
        'savefig.pad_inches': .02, 'savefig.dpi': 600, 'pdf.fonttype': 42, 'ps.fonttype': 42})
    return size


def save(fig, stem):
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    for extension in ['pdf', 'png']:
        fig.savefig(stem.with_suffix('.'+extension), dpi=600)


def panel_label(ax, label, x=-.1, y=1.05, **kwargs):
    return ax.text(x, y, label, transform=ax.transAxes, **kwargs)
