import sys as _runtime_sys
from pathlib import Path as _RuntimePath
_runtime_sys.path.insert(0, str(_RuntimePath(__file__).resolve().parents[2]))
from audit_runtime import PACKAGE_ROOT, WORKSPACE, record_path, require_records
import sys
from pathlib import Path
import matplotlib as mpl
import figstyle
WIDTH_CM = 12.2
WIDTH_IN = WIDTH_CM / 2.54
METHODS = {'elapsed': dict(label='Elapsed-matched (n=1,518)', color='#D55E00', marker='^', mfc='#D55E00'), 'full': dict(label='Full window', color='#0072B2', marker='o', mfc='#0072B2'), 'cap30': dict(label='30-minute cap', color='#333333', marker='s', mfc='white'), 'time': dict(label='Chronological (primary)', color='#0072B2', marker='o', mfc='#0072B2'), 'group': dict(label='Grouped', color='#333333', marker='D', mfc='white'), 'recent': dict(label='Recent target change', color='#0072B2', marker='o', mfc='#0072B2'), 'no_recent': dict(label='No recent target change', color='#333333', marker='s', mfc='white')}
UNITS = {'excess': 'Excess co-occurrence probability', 'accuracy': 'Accuracy difference (percentage points)', 'recall': 'Event recall (hits / eligible events)'}

def apply(height=3.3):
    figstyle.apply(width=WIDTH_IN, aspect=height / WIDTH_IN)
    mpl.rcParams.update({'axes.labelsize': 9, 'figure.constrained_layout.use': False, 'savefig.bbox': None, 'savefig.pad_inches': 0, 'font.serif': ['Liberation Serif', 'STIXGeneral'], 'axes.spines.top': False, 'axes.spines.right': False})

def save(fig, stem):
    for extension in ('pdf', 'png'):
        fig.savefig(Path(stem).with_suffix('.' + extension), dpi=600, metadata={'Creator': 'Matplotlib'} if extension == 'pdf' else None)

def style(key):
    return dict(METHODS[key])
