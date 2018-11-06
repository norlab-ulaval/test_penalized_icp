import numpy as np

from pypm.data_points import DataPoints
import pytest

A_ROW_VECTOR = np.array([[0, 0, 0]], dtype=np.float64)

A_VALID_INTEGER_MATRIX = np.array([[1, 1],
                                   [2, 2],
                                   [3, 3]])
A_VALID_MATRIX = np.array([[1, 1],
                           [2, 2],
                           [3, 3]], dtype=np.float64)


def test_from_numpy_given_a_row_vector_then_exception_is_throw():
    with pytest.raises(RuntimeError):
        DataPoints.from_numpy(A_ROW_VECTOR)


def test_from_numpy_given_a_valid_matrix_integer_format_then_its_converted():
    DataPoints.from_numpy(A_VALID_INTEGER_MATRIX, make_homogeneous=False)


def test_given_a_valid_matrix_then_from_numpy_and_to_numpy_should_be_equal():
    dp = DataPoints.from_numpy(A_VALID_MATRIX, make_homogeneous=False)
    assert (A_VALID_MATRIX == dp.numpy).all()


