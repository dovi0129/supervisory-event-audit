import sys as _runtime_sys
from pathlib import Path as _RuntimePath
_runtime_sys.path.insert(0, str(_RuntimePath(__file__).resolve().parents[2]))
from audit_runtime import PACKAGE_ROOT, WORKSPACE, record_path, require_records
if __name__ == "__main__":
    require_records()
import sys
sys.path.insert(0, str(PACKAGE_ROOT / 'scripts/paper1_figs'))
import figstyle
apply = figstyle.apply
save = figstyle.save
panel_label = figstyle.panel_label
METHODS = {'ours': dict(label='Proposed', color='#D55E00', ls='-', marker='o'), 'baseline': dict(label='Baseline', color='#0072B2', ls='--', marker='s'), 'ablation': dict(label='w/o X', color='#009E73', ls='-.', marker='^'), 'ref': dict(label='Reference', color='#555555', ls=':', marker=None)}
UNITS = {'time': 'Time (h)', 'error': 'Tracking error (%)'}

def style(method, **override):
    d = dict(METHODS[method])
    d.update(override)
    return d
