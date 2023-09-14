"""Test stats functions."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from skfda import FDataBasis, FDataGrid
from skfda.datasets import make_gaussian_process
from skfda.exploratory.stats import std
from skfda.misc.covariances import Gaussian
from skfda.representation.basis import (
    FourierBasis,
    MonomialBasis,
    TensorBasis,
    VectorValuedBasis,
)
from skfda.typing._numpy import NDArrayFloat


@pytest.fixture(params=[61, 71])
def n_basis(request: Any) -> int:
    """Fixture for n_basis to test."""
    return request.param


@pytest.fixture
def start() -> int:
    """Fixture for the infimum of the domain."""
    return 0


@pytest.fixture
def stop() -> int:
    """Fixture for the supremum of the domain."""
    return 1


@pytest.fixture
def n_features() -> int:
    """Fixture for the number of features."""
    return 1000


@pytest.fixture
def gaussian_process(start: int, stop: int, n_features: int) -> FDataGrid:
    """Fixture for a Gaussian process."""
    return make_gaussian_process(
        start=start,
        stop=stop,
        n_samples=100,
        n_features=n_features,
        mean=0.0,
        cov=Gaussian(variance=1, length_scale=0.1),
        random_state=0,
    )


def test_std_gaussian_fourier(
    start: int,
    stop: int,
    n_features: int,
    n_basis: int,
    gaussian_process: FDataGrid,
) -> None:
    """Test standard deviation: Gaussian processes and a Fourier basis."""
    fourier_basis = FourierBasis(n_basis=n_basis, domain_range=(0, 1))
    fd = gaussian_process.to_basis(fourier_basis)

    std_fd = std(fd)
    grid = np.linspace(start, stop, n_features)
    almost_std_fd = std(fd.to_grid(grid)).to_basis(fourier_basis)

    inner_grid_limit = n_features // 10
    inner_grid = grid[inner_grid_limit:-inner_grid_limit]
    np.testing.assert_allclose(
        std_fd(inner_grid),
        almost_std_fd(inner_grid),
        rtol=1e-3,
    )

    outer_grid = grid[:inner_grid_limit] + grid[-inner_grid_limit:]
    np.testing.assert_allclose(
        std_fd(outer_grid),
        almost_std_fd(outer_grid),
        rtol=1e-2,
    )


@pytest.mark.parametrize("fdatagrid, expected_std_data_matrix", [
    (
        FDataGrid(
            data_matrix=[
                [[0, 1, 2, 3, 4, 5], [0, -1, -2, -3, -4, -5]],
                [[2, 3, 4, 5, 6, 7], [-2, -3, -4, -5, -6, -7]],
            ],
            grid_points=[
                [-2, -1],
                [0, 1, 2, 3, 4, 5],
            ],
        ),
        np.full((1, 2, 6, 1), np.sqrt(2)),
    ),
    (
        FDataGrid(
            data_matrix=[
                [
                    [[10, 11], [10, 12], [11, 14]],
                    [[15, 16], [12, 15], [20, 13]],
                ],
                [
                    [[11, 12], [11, 13], [12, 13]],
                    [[14, 15], [11, 16], [21, 12]],
                ],
            ],
            grid_points=[
                [0, 1],
                [0, 1, 2],
            ],
        ),
        np.full((1, 2, 3, 2), np.sqrt(1 / 2)),
    ),
])
def test_std_fdatagrid(
    fdatagrid: FDataGrid,
    expected_std_data_matrix: NDArrayFloat,
) -> None:
    """Test some std_fdatagrid cases."""
    np.testing.assert_allclose(
        std(fdatagrid).data_matrix,
        expected_std_data_matrix,
    )


@pytest.mark.parametrize("fdatabasis, expected_std_coefficients", [
    (
        FDataBasis(
            basis=VectorValuedBasis([
                MonomialBasis(domain_range=(0, 1), n_basis=3),
                MonomialBasis(domain_range=(0, 1), n_basis=3),
            ]),
            coefficients=[
                [0, 0, 0, 0, 0, 0],
                [1, 0, 0, 1, 0, 0],
            ],
        ),
        np.array([[np.sqrt(1 / 2), 0, 0, np.sqrt(1 / 2), 0, 0]]),
    ),
    (
        FDataBasis(
            basis=VectorValuedBasis([
                FourierBasis(domain_range=(0, 1), n_basis=5),
                MonomialBasis(domain_range=(0, 1), n_basis=4),
            ]),
            coefficients=[
                [0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 1, 0, 0, 0],
            ],
        ),
        np.array([[np.sqrt(1 / 2), 0, 0, 0, 0, np.sqrt(1 / 2), 0, 0, 0]]),
    ),
    (
        FDataBasis(
            basis=TensorBasis([
                MonomialBasis(domain_range=(0, 1), n_basis=4),
                MonomialBasis(domain_range=(0, 1), n_basis=4),
            ]),
            coefficients=[
                np.zeros(16),
                np.pad([1], (0, 15)),
            ],
        ),
        [np.pad([np.sqrt(1 / 2)], (0, 15))],
    ),
    (
        FDataBasis(
            basis=VectorValuedBasis([
                TensorBasis([
                    MonomialBasis(domain_range=(0, 1), n_basis=2),
                    MonomialBasis(domain_range=(0, 1), n_basis=2),
                ]),
                TensorBasis([
                    MonomialBasis(domain_range=(0, 1), n_basis=2),
                    MonomialBasis(domain_range=(0, 1), n_basis=2),
                ]),
            ]),
            coefficients=[
                [0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 1, 0, 0, 0],
            ],
        ),
        np.array([[np.sqrt(1 / 2), 0, 0, 0] * 2]),
    ),
])
def test_std_fdatabasis(
    fdatabasis: FDataBasis,
    expected_std_coefficients: NDArrayFloat,
) -> None:
    """Test some std_fdatabasis cases."""
    np.testing.assert_allclose(
        std(fdatabasis).coefficients,
        expected_std_coefficients,
        rtol=1e-7,
        atol=1e-7,
    )
