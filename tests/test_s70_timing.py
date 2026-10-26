"""Known-answer timing tests, independent of confidential process data."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts/s70'))
from s70_features import causal_features, future_labels  # noqa: E402


def sample():
    idx = pd.date_range('2022-01-01', periods=130, freq='min')
    o = pd.DataFrame(index=idx)
    for name in ['moisture', 'size_moisture', 'weight']:
        o[name] = np.arange(130, dtype=float)
        o[name + '_target'] = 1.
    for name in ['speed', 'pressure_after', 'pressure_pre']:
        o[name] = np.arange(130, dtype=float)
    o['grade'] = ['grade 1'] * 65 + ['grade 2'] * 65
    o['mode_00'] = 1.
    return o


def test_future_values_are_invisible():
    o = sample()
    baseline = causal_features(o)
    for boundary in [0, 1, 59, 60, 65, 90, 128]:
        altered = o.copy()
        for c in altered:
            altered.loc[altered.index[boundary + 1]:, c] = 'grade 99' if c == 'grade' else 9000.
        pd.testing.assert_frame_equal(baseline.iloc[:boundary + 1], causal_features(altered).iloc[:boundary + 1])


def test_label_boundaries():
    e = np.zeros(12, bool)
    e[6] = True
    np.testing.assert_array_equal(np.where(future_labels(e, 5))[0], [1, 2, 3, 4, 5])
    np.testing.assert_array_equal(np.where(future_labels(e, 1))[0], [5])
    assert not future_labels(e, 5)[6]
    e[-1] = True
    assert future_labels(e, 1)[-2]
    assert not future_labels(e, 1)[-1]


def test_trailing_centering_known_answer():
    x = causal_features(sample())
    assert x['weight_error'].iloc[60] == 29.5
    assert x['moisture_slope5'].iloc[6] == 1.
    assert pd.isna(x['grade_age'].iloc[64])
    assert x['grade_age'].iloc[65] == 0.
    assert x['grade_age'].iloc[66] == 1.


def test_split_purge_and_event_metric_known_answers():
    from s70_fit import complete_windows, event_hits
    mask = np.ones(15, bool)
    mask[7] = False
    kept = complete_windows(mask, 2, history=2)
    np.testing.assert_array_equal(np.flatnonzero(kept), [2, 3, 4, 10, 11, 12])
    alarm = np.zeros(20, bool)
    alarm[[5, 12]] = True
    domain = np.ones(20, bool)
    positions, hits, leads = event_hits(alarm, [5, 8, 18], domain, 3)
    np.testing.assert_array_equal(positions, [5, 8, 18])
    np.testing.assert_array_equal(hits, [False, True, False])
    assert leads == [3]
    domain[5:8] = False
    positions, hits, leads = event_hits(alarm, [8], domain, 3)
    assert len(positions) == 0


def test_v02_lot_edges_alarms_and_verdict():
    from s70_fit import lot_windows, rising_alarms, metrics
    from s70_report import category, mechanical
    np.testing.assert_array_equal(lot_windows(['a', 'a', 'b', 'a', 'a'], 1), [True, False, False, True, False])
    score = np.zeros(15)
    score[[1, 3, 5, 8, 11, 13]] = 1.
    episode = np.ones(15)
    episode[13:] = 2
    alarm = rising_alarms(score, .5, np.ones(15, bool), 3, episode)
    np.testing.assert_array_equal(np.flatnonzero(alarm), [1, 5, 11, 13])
    assert category(.6, .2, 1/3) == '부분'
    assert category(.5, .3, .25) == '설명 가능'
    assert mechanical({'event_recall': .5, 'minute_precision': .3, 'chance_recall': .25,
                       'chance_excess': 1/3, 'events': 10, 'event_hits': 5})['one_event_boundary']
    y = future_labels(np.isin(np.arange(15), [6, 12]), 3)
    row = metrics(y, score, alarm, [6, 12], np.ones(15, bool), 3, np.random.default_rng(1))
    assert row['events'] == 2 and row['event_hits'] == 2
    assert row['chance_valid_shifts'] == 50
    assert abs(row['chance_excess'] - (row['event_recall']-row['chance_recall'])/(1-row['chance_recall'])) < 1e-12


def test_censoring_does_not_create_rising_edges():
    from s70_fit import rising_alarms
    score = np.ones(20)
    mode = np.ones(20, bool)
    evaluation = mode.copy()
    evaluation[4:10] = False
    alarms = rising_alarms(score, .5, mode, 3, np.ones(20))
    # Removing a LOT label boundary must not introduce an onset at minute 10.
    np.testing.assert_array_equal(np.flatnonzero(alarms & evaluation), [0])
