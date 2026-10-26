"""Small invented event stream; no plant records or aggregate rows are sampled."""
import json
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts/s70'))
sys.path.insert(0, str(ROOT / 'scripts/s77'))
from s70_features import future_labels  # noqa: E402
from s70_fit import rising_alarms, event_hits  # noqa: E402
from s77_common import lot_layout, rotated_indices  # noqa: E402


def demonstrate():
    events = np.zeros(20, bool)
    events[[6, 14]] = True
    score = np.zeros(20)
    score[[3, 5, 11]] = 1
    alarms = rising_alarms(score, .5, np.ones(20, bool), 5, np.ones(20))
    _, hits, leads = event_hits(alarms, [6, 14], np.ones(20, bool), 5)
    donor = rotated_indices(lot_layout(['A']*10+['B']*10), np.random.default_rng(4))
    shifted = events[donor]
    assert hits.tolist() == [True, True]
    assert np.flatnonzero(alarms).tolist() == [3, 11]
    assert shifted[:10].sum() == events[:10].sum()
    assert shifted[10:].sum() == events[10:].sum()
    return {'data': 'fully synthetic', 'alarms': np.flatnonzero(alarms).tolist(),
            'hits': hits.tolist(), 'lead_minutes': leads,
            'future_label_positive_minutes': np.flatnonzero(future_labels(events, 5)).tolist(),
            'within_group_event_counts_preserved': True}


if __name__ == '__main__':
    print(json.dumps(demonstrate(), indent=2))
