import sys as _runtime_sys
from pathlib import Path as _RuntimePath
_runtime_sys.path.insert(0, str(_RuntimePath(__file__).resolve().parents[2]))
from audit_runtime import PACKAGE_ROOT, WORKSPACE, record_path, require_records
import numpy as np
from s74_episodes import first_confirmation, endpoint, edit_distance, variance_parts
from s74_recipe import distance_matrix

def test_endpoint():
    assert first_confirmation([True] * 14) is None
    assert first_confirmation([True] * 15) == 14
    assert first_confirmation([True] * 10 + [False] + [True] * 15) == 25
    assert endpoint(0, 15, np.ones(20, bool)) == (15, 'stable_band', 0)
    assert endpoint(0, 14, np.ones(20, bool)) == (14, None, None)
    assert endpoint(3, 20, np.ones(20, bool)) == (18, 'stable_band', 3)

def test_edit_and_variance():
    assert edit_distance(('a', 'b'), ('b', 'a')) == 2
    assert edit_distance((), ('a',)) == 1
    a = np.array([[0, 1], [1, 0], [-1, -1]])
    raw, norm = distance_matrix(a, np.array([2, 2, 0]))
    np.testing.assert_allclose(raw, [[0, 2, 2], [2, 0, 2], [2, 2, 0]])
    np.testing.assert_allclose(norm, [[0, 1, 1], [1, 0, 1], [1, 1, 0]])
    result = variance_parts([0, 2, 4, 6], ['a', 'a', 'b', 'b'])
    assert abs(result['within'] - 1) < 1e-12
    assert abs(result['between'] - 4) < 1e-12
    assert abs(result['ratio'] - 0.25) < 1e-12
