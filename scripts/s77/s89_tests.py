import sys as _runtime_sys
from pathlib import Path as _RuntimePath
_runtime_sys.path.insert(0, str(_RuntimePath(__file__).resolve().parents[2]))
from audit_runtime import PACKAGE_ROOT, WORKSPACE, record_path, require_records
import numpy as np
import pandas as pd
from s89_q2 import frequency_predictions, history_columns, grouped_statistics, measures
from s89_controls import transition_anchor, contiguous_true_blocks, rotate_transition_only, recent_target_signal, matched_donor_pool

def test_markov_train_only_and_backoff():
    f = pd.DataFrame({'pair': ['a', 'a', 'b', 'a', 'new'], 'hist__last_kind': ['none', 'x', 'x', 'x', 'unseen'], 'kind_code': [0, 1, 2, 8, 8]})
    train = np.array([1, 1, 1, 0, 0], bool)
    plain, prior, _ = frequency_predictions(f, train, False)
    paired, _, _ = frequency_predictions(f, train, True)
    assert paired[3].argmax() == 1
    assert np.isclose(plain[3, 1], plain[3, 2])
    np.testing.assert_allclose(plain[4], prior)
    f.loc[~train, 'kind_code'] = 0
    changed, _, _ = frequency_predictions(f, train, True)
    np.testing.assert_allclose(changed, paired)
    np.testing.assert_allclose(paired.sum(1), 1)

def test_history_filter_and_metric_weights():
    f = pd.DataFrame(columns=['recipe__pair', 'hist__last_kind', 'hist__count_0', 'hist__target_0', 'hist__target_age_0', 'hist__mode_00', 'state__x'])
    assert history_columns(f) == ['recipe__pair', 'hist__last_kind', 'hist__count_0']
    y = np.array([0, 1, 1])
    pred = np.array([0, 0, 1])
    p = np.full((3, 9), 0.01)
    p[np.arange(3), pred] = 0.92
    count, cm, ll = grouped_statistics(y, p, ['a', 'a', 'b'])
    ordinary = measures(cm.sum(0), ll.sum(), count.sum())
    equal_episode = measures((cm / count[:, None, None]).sum(0), (ll / count).sum(), 2.0)
    assert np.isclose(ordinary['accuracy'], 2 / 3)
    assert np.isclose(equal_episode['accuracy'], 0.75)
    assert np.isclose(ordinary['balanced_accuracy'], 0.75)
    assert np.isclose(ordinary['macro_f1'], (2 / 3 + 2 / 3) / 9)

def test_signed_anchor_boundary_and_donors():
    np.testing.assert_array_equal(transition_anchor([69, 70, 100, 150, 161, 170, 201, 261], [100, 200]), [-1, 0, 0, 0, -1, 1, 1, -1])
    centers = np.array([50, 150, 250])
    region = np.full(400, '지종 전환', object)
    runs = np.arange(400) // 100
    candidates, valid = matched_donor_pool(55, 20, 0, centers, region, runs)
    assert not valid[0].any() and valid[1:].all()
    assert ((candidates[valid] >= 150) & (candidates[valid] <= 260)).all()

def test_rotation_preserves_joint_union_and_outside():
    flag = np.array([1, 0, 1, 1, 0, 0, 1, 0, 1], bool)
    mask = np.array([0, 1, 1, 1, 0, 1, 1, 0, 0], bool)
    blocks = contiguous_true_blocks(mask)
    moved = rotate_transition_only(flag, blocks, np.random.default_rng(7))
    np.testing.assert_array_equal(moved[~mask], flag[~mask])
    for start, end in blocks:
        assert moved[start:end].sum() == flag[start:end].sum()

def test_recent_target_uses_current_and_no_future():
    f = pd.DataFrame({c: [np.nan, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0] for c in ['moisture_target_age', 'size_moisture_target_age', 'weight_target_age']})
    np.testing.assert_array_equal(recent_target_signal(f, 3), [0, 1, 1, 1, 0, 0, 0])
    before = recent_target_signal(f, 5)
    f.iloc[5:] = 0
    np.testing.assert_array_equal(recent_target_signal(f, 5)[:5], before[:5])
