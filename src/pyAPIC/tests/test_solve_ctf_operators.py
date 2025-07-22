import numpy as np

from pyAPIC.core.solve_ctf_operators import (
    build_row_map,
    check_full_column_rank,
    form_diff_and_offset_operator,
    form_Zernike_operator,
)


def test_check_full_column_rank():
    A = np.eye(3)
    assert check_full_column_rank(A)
    B = np.array([[1, 2], [2, 4]])
    assert not check_full_column_rank(B)


def test_build_row_map():
    K = np.array([[0, 0], [1, 0], [0, 1]])
    mapping = build_row_map(K)
    assert mapping[(0, 0)] == 0
    assert mapping[(1, 0)] == 1
    assert mapping[(0, 1)] == 2


def test_form_diff_and_offset_operator():
    K = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
    idx_overlap = np.array([3])
    ki = np.array([1, 0])
    kl = np.array([0, 1])
    row_map = build_row_map(K)
    D, D0 = form_diff_and_offset_operator(K, idx_overlap, ki, kl, row_map)
    assert D.shape == (1, 4)
    assert D0.shape == (1, 4)
    assert np.array_equal(D.toarray(), [[0, -1, 1, 0]])
    assert np.array_equal(D0.toarray(), [[0, 1, -1, 0]])


def test_form_zernike_operator_shape():
    Hz = form_Zernike_operator([3, 4], 4)
    assert Hz.shape == (16, 2)
