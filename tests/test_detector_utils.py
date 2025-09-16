import numpy as np
import os
from src import detector_pipeline as dp


def test_l2_normalize_zero_vector():
    mat = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32)
    out = dp.l2_normalize(mat)
    # zero row remains zero, other row normalized
    assert out.shape == mat.shape
    assert np.allclose(out[0], np.zeros(3))
    assert np.isclose(np.linalg.norm(out[1]), 1.0)


def test_lpips_distance_not_installed_returns_none():
    # This test assumes lpips not installed in the test environment; if installed, skip
    if dp._HAS_LPIPS:
        import pytest
        pytest.skip("LPIPS is installed in this env; skipping behavior test")
    a = os.path.join(os.path.dirname(__file__), 'fixtures', 'a.jpg')
    b = os.path.join(os.path.dirname(__file__), 'fixtures', 'b.jpg')
    # Missing files or missing lpips -> should return None
    res = dp.lpips_distance(a, b)
    assert res is None
