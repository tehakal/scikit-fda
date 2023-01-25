"""Abstract base class for basis."""

from __future__ import annotations

from typing import Any, Tuple, TypeVar

import multimethod
import numpy as np

from ...typing._numpy import NDArrayFloat
from .._functional_data import FData
from ..grid import FDataGrid
from ._basis import Basis
from ._fdatabasis import FDataBasis

T = TypeVar("T", bound="CustomBasis")


class CustomBasis(Basis):
    """Basis composed of custom functions.

    Defines a basis composed of the functions in the :class: `FData` object
    passed as argument.
    The functions must be linearly independent, otherwise
    an exception is raised.

    Parameters:
        fdata: Functions that define the basis.

    """

    def __init__(
        self,
        *,
        fdata: FData,
    ) -> None:
        """Basis constructor."""
        super().__init__(
            domain_range=fdata.domain_range,
            n_basis=fdata.n_samples,
        )
        self._check_linearly_independent(fdata)

        self.fdata = fdata

    @multimethod.multidispatch
    def _check_linearly_independent(self, fdata: FData) -> None:
        raise ValueError("Unexpected type of functional data object.")

    @_check_linearly_independent.register
    def _check_linearly_independent_grid(self, fdata: FDataGrid) -> None:
        # Reshape to a bidimensional matrix. This only affects FDataGrids
        # whose codomain is not 1-dimensional and it can be done because
        # checking linear independence in (R^n)^k is equivalent to doing
        # it in R^(nk).
        coord_matrix = fdata.data_matrix.reshape(
            fdata.data_matrix.shape[0],
            -1,
        )
        return self._check_linearly_independent_matrix(coord_matrix)

    @_check_linearly_independent.register
    def _check_linearly_independent_basis(self, fdata: FDataBasis) -> None:
        return self._check_linearly_independent_matrix(fdata.coefficients)

    def _check_linearly_independent_matrix(self, matrix: NDArrayFloat) -> None:
        """Check if the functions are linearly independent."""
        if matrix.shape[0] > matrix.shape[1]:
            raise ValueError(
                "There are more functions than the maximum dimension of the "
                "space that they could generate.",
            )

        rank = np.linalg.matrix_rank(matrix)
        if rank != matrix.shape[0]:
            raise ValueError(
                "There are only {rank} linearly independent "
                "functions".format(
                    rank=rank,
                ),
            )

    def _derivative_basis_and_coefs(
        self: T,
        coefs: NDArrayFloat,
        order: int = 1,
    ) -> Tuple[T, NDArrayFloat]:
        deriv_fdata = self.fdata.derivative(order=order)

        return self._create_subspace_basis_coef(deriv_fdata, coefs)

    @multimethod.multidispatch
    def _create_subspace_basis_coef(
        self: T,
        fdata: FData,
        coefs: np.ndarray,
    ) -> Tuple[T, NDArrayFloat]:
        """
        Create a basis of the subspace generated by the given functions.

        Args:
            fdata: The resulting basis will span the subspace generated
                by these functions.
            coefs: Coefficients of some functions in the given fdata.
                These coefficients will be transformed into the coefficients
                of the same functions in the resulting basis.
        """
        raise ValueError(
            "Unexpected type of functional data object: {type}.".format(
                type=type(fdata),
            ),
        )

    @_create_subspace_basis_coef.register
    def _create_subspace_basis_coef_grid(
        self: T,
        fdata: FDataGrid,
        coefs: np.ndarray,
    ) -> Tuple[T, NDArrayFloat]:

        # Reshape to a bidimensional matrix. This can be done because
        # working in (R^n)^k is equivalent to working in R^(nk) when
        # it comes to linear independence and basis.
        data_matrix_reshaped = fdata.data_matrix.reshape(
            fdata.data_matrix.shape[0],
            -1,
        )
        # If the basis formed by the derivatives has maximum rank,
        # we can just return that
        rank = np.linalg.matrix_rank(data_matrix_reshaped)
        if rank == fdata.n_samples:
            return type(self)(fdata=fdata), coefs

        # Otherwise, we need to find the basis of the subspace generated
        # by the functions
        q, r = np.linalg.qr(data_matrix_reshaped.T)

        # Go back from R^(nk) to (R^n)^k
        fdata.data_matrix = q.T.reshape(
            -1,
            *fdata.data_matrix.shape[1:],
        )

        new_basis = type(self)(fdata=fdata)

        # Since the QR decomponsition yields an orthonormal basis,
        # the coefficients are just the projections of values of
        # the functions in every point (coefs @ data_matrix_reshaped)
        # in the new basis (q).
        # Note that to simply the calculations, we use both the data_matrix
        # and the basis matrix in R^(nk) instead of the original space
        values_in_eval_points = coefs @ data_matrix_reshaped
        coefs = values_in_eval_points @ q

        return new_basis, coefs

    @_create_subspace_basis_coef.register
    def _create_subspace_basis_coef_basis(
        self: T,
        fdata: FDataBasis,
        coefs: np.ndarray,
    ) -> Tuple[T, NDArrayFloat]:

        # If the basis formed by the derivatives has maximum rank,
        # we can just return that
        rank = np.linalg.matrix_rank(fdata.coefficients)
        if rank == fdata.n_samples:
            return type(self)(fdata=fdata), coefs

        q, r = np.linalg.qr(fdata.coefficients.T)

        fdata.coefficients = q.T

        new_basis = type(self)(fdata=fdata)

        # Since the QR decomponsition yields an orthonormal basis,
        # the coefficients are just the result of projecting the
        # coefficients in the underlying basis of the FDataBasis onto
        # the new basis (q)
        coefs_wrt_underlying_fdata_basis = coefs @ fdata.coefficients
        coefs = coefs_wrt_underlying_fdata_basis @ q

        return new_basis, coefs

    def _coordinate_nonfull(
        self,
        coefs: NDArrayFloat,
        key: int | slice,
    ) -> Tuple[Basis, NDArrayFloat]:
        return CustomBasis(fdata=self.fdata.coordinates[key]), coefs

    def _evaluate(
        self,
        eval_points: NDArrayFloat,
    ) -> NDArrayFloat:
        return self.fdata(eval_points)

    def _gram_matrix(self) -> NDArrayFloat:
        """
        Compute the Gram matrix.

        Subclasses may override this method for improving computation
        of the Gram matrix.

        """
        if isinstance(self.fdata, FDataBasis):
            basis_gram = self.fdata.basis.gram_matrix()
            coefficients = self.fdata.coefficients
            return coefficients @ basis_gram @ coefficients.T

        # TODO: It would be better to call inner_product_matrix but that
        # introduces a circular dependency
        gram_matrix = np.empty((self.n_basis, self.n_basis))
        for i in range(self.n_basis):
            for j in range(i, self.n_basis):
                gram_matrix[i, j] = (self.fdata[i] * self.fdata[j]).integrate()
                gram_matrix[j, i] = gram_matrix[i, j]
        return gram_matrix

    def __len__(self) -> int:
        return self.n_basis

    @property
    def dim_codomain(self) -> int:
        return self.fdata.dim_codomain

    def __eq__(self, other: Any) -> bool:
        return super().__eq__(other) and all(self.fdata == other.fdata)

    def __ne__(self, other: Any) -> bool:
        return not self.__eq__(other)

    def __repr__(self) -> str:
        """Representation of a CustomBasis object."""
        return "{super}, fdata={fdata}".format(
            super=super().__repr__(),
            fdata=self.fdata,
        )

    def __hash__(self) -> int:
        return hash(self.fdata)
