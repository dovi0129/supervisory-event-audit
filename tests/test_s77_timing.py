"""S77 timing, availability, censoring, joint shift and exchange known answers."""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/s77"))
from s77_common import (
    interval_any,
    lot_layout,
    rotated_indices,
    exchange_indices,
    kaplan_meier,
)  # noqa: E402
from s77_q3 import past_event_features  # noqa: E402


def test_joint_shift_and_label_boundary():
    matrix = np.array([[1, 1, 0], [0, 0, 1], [0, 0, 0], [1, 0, 1]], bool)
    donor = rotated_indices(lot_layout(["a", "a", "b", "b"]), np.random.default_rng(1))
    moved = matrix[donor]
    np.testing.assert_array_equal(matrix[:2].sum(axis=0), moved[:2].sum(axis=0))
    np.testing.assert_array_equal(matrix[2:].sum(axis=0), moved[2:].sum(axis=0))
    assert (matrix[:, 0] & matrix[:, 1]).sum() == (moved[:, 0] & moved[:, 1]).sum()
    flag = np.zeros(12, bool)
    flag[6] = True
    assert interval_any(flag, [5], [7])[0]
    assert not interval_any(flag, [6], [6])[0]
    assert not interval_any(flag, [0], [6])[0]


def test_detector_future_invariance():
    target = np.zeros(40, bool)
    mode = target.copy()
    target[2] = True
    mode[4] = True
    a = past_event_features(target, mode)
    target[21:] = True
    mode[21:] = True
    pd.testing.assert_frame_equal(
        a.iloc[:21], past_event_features(target, mode).iloc[:21]
    )


def test_exchange_and_survival():
    f = pd.DataFrame(
        {
            "episode_id": ["a", "b", "a"],
            "pair": ["x", "x", "y"],
            "step": [1, 1, 1],
            "elapsed_bin": [0, 0, 0],
        }
    )
    donor, changed = exchange_indices(f, np.random.default_rng(1))
    np.testing.assert_array_equal(donor, [1, 0, 2])
    np.testing.assert_array_equal(changed, [True, True, False])
    rows, median = kaplan_meier([1, 2, 3], [True, False, True])
    assert median == 3
    assert abs(rows[0]["survival"] - 2 / 3) < 1e-12


def test_bundle_completion_and_censoring():
    from s77_q2_build import decision_records, label_history

    bundles = [
        {
            "start": 0,
            "complete": 1,
            "members": [
                {"kind": "target", "loop": "A", "delta": 100.0, "dur_min": 2},
                {"kind": "mode", "loop": "M", "delta": -1.0, "dur_min": 1},
            ],
        },
        {
            "start": 3,
            "complete": 3,
            "members": [{"kind": "target", "loop": "A", "delta": 2.0, "dur_min": 1}],
        },
        {
            "start": 4,
            "complete": 5,
            "members": [{"kind": "target", "loop": "A", "delta": 3.0, "dur_min": 2}],
        },
    ]
    rows = decision_records(bundles, 5)
    assert rows[0]["completed"] == [] and rows[0]["next_bundle"] == 1
    assert rows[1]["relative_min"] == 1 and rows[1]["completed"] == [0]
    assert rows[2]["censored"] and rows[2]["censor_reason"] == "incomplete_next_bundle"
    import json

    frame = pd.DataFrame(
        {
            "completed_bundle_ids": [json.dumps(r["completed"]) for r in rows],
            "next_bundle_id": [
                r["next_bundle"] if r["next_bundle"] is not None else -1 for r in rows
            ],
            "relative_min": [r["relative_min"] for r in rows],
        }
    )
    labeled = label_history(frame, bundles, {"A": 1.0}, ["A"])
    assert labeled["hist__last_size"].iloc[0] == 0.0
    assert labeled["hist__last_size"].iloc[1] == 100.0
    assert labeled.label_kind.iloc[-1] is None
    assert set(labeled.label_kind.dropna()).issubset(
        {"A", "mode", "manual_output", "DCS"}
    )


def test_measurement_features_ignore_future():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/s70"))
    from s70_features import causal_features
    from test_s70_timing import sample
    from s77_q2_build import MEASUREMENTS

    raw = sample()
    base = causal_features(raw)
    for c in raw:
        raw.loc[raw.index[81] :, c] = "grade 99" if c == "grade" else 9999.0
    pd.testing.assert_frame_equal(
        base[MEASUREMENTS].iloc[:81], causal_features(raw)[MEASUREMENTS].iloc[:81]
    )
