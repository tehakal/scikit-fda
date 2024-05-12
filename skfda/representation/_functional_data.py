"""Module for functional data manipulation.

Defines the abstract class that should be implemented by the funtional data
objects of the package and contains some commons methods.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, TypeVar

import numpy as np
import pandas
from typing_extensions import override

from skfda._utils.ndfunction._region import AxisAlignedBox

from .._utils.ndfunction import NDFunction, concatenate as concatenate
from .._utils.ndfunction._array_api import Array, DType, numpy_namespace
from .._utils.ndfunction._region import Region
from .._utils.ndfunction.utils.validation import check_grid_points
from ..typing._base import DomainRange, LabelTuple, LabelTupleLike
from .extrapolation import ExtrapolationLike

InputDType = TypeVar('InputDType', bound=DType)
OutputDType = TypeVar('OutputDType', bound=DType)


class FData(  # noqa: WPS214
    ABC,
    NDFunction,
    pandas.api.extensions.ExtensionArray,  # type: ignore[misc]
):
    """
    Defines the structure of a functional data object.

    Attributes:
        n_samples: Number of samples.
        dim_domain: Dimension of the domain.
        dim_codomain: Dimension of the image.
        extrapolation: Default extrapolation mode.
        dataset_name: Name of the dataset.
        argument_names: Tuple containing the names of the different
            arguments.
        coordinate_names: Tuple containing the names of the different
            coordinate functions.

    """

    dataset_name: str | None

    def __init__(
        self,
        *,
        extrapolation: ExtrapolationLike | None = None,
        dataset_name: str | None = None,
        argument_names: LabelTupleLike | None = None,
        coordinate_names: LabelTupleLike | None = None,
        sample_names: LabelTupleLike | None = None,
    ) -> None:

        self.extrapolation = extrapolation  # type: ignore[assignment]
        self.dataset_name = dataset_name

        self.argument_names = argument_names  # type: ignore[assignment]
        self.coordinate_names = coordinate_names  # type: ignore[assignment]
        self.sample_names = sample_names  # type: ignore[assignment]

    @override
    @property
    def array_backend(self) -> Any:
        return numpy_namespace

    @property
    def argument_names(self) -> LabelTuple:
        return self._argument_names

    @argument_names.setter
    def argument_names(
        self,
        names: LabelTupleLike | None,
    ) -> None:
        if names is None:
            names = (None,) * self.dim_domain
        else:
            names = tuple(names)
            if len(names) != self.dim_domain:
                raise ValueError(
                    "There must be a name for each of the "
                    "dimensions of the domain.",
                )

        self._argument_names = names

    @property
    def coordinate_names(self) -> LabelTuple:
        return self._coordinate_names

    @coordinate_names.setter
    def coordinate_names(
        self,
        names: LabelTupleLike | None,
    ) -> None:
        if names is None:
            names = (None,) * self.dim_codomain
        else:
            names = tuple(names)
            if len(names) != self.dim_codomain:
                raise ValueError(
                    "There must be a name for each of the "
                    "dimensions of the codomain.",
                )

        self._coordinate_names = names

    @property
    def sample_names(self) -> LabelTuple:
        return self._sample_names

    @sample_names.setter
    def sample_names(self, names: LabelTupleLike | None) -> None:
        if names is None:
            names = (None,) * self.n_samples
        else:
            names = tuple(names)
            if len(names) != self.n_samples:
                raise ValueError(
                    "There must be a name for each of the samples.",
                )

        self._sample_names = names

    @property
    @abstractmethod
    def n_samples(self) -> int:
        """Return the number of samples.

        Returns:
            Number of samples of the FData object.

        """
        pass

    @override
    @property
    def shape(self) -> tuple[int, ...]:
        return (self.n_samples,)

    @property
    @abstractmethod
    def dim_domain(self) -> int:
        """Return number of dimensions of the :term:`domain`.

        Returns:
            Number of dimensions of the domain.

        """
        pass

    @override
    @property
    def input_shape(self) -> tuple[int, ...]:
        return (self.dim_domain,)

    @property
    @abstractmethod
    def dim_codomain(self) -> int:
        """Return number of dimensions of the :term:`codomain`.

        Returns:
            Number of dimensions of the codomain.

        """
        pass

    @override
    @property
    def output_shape(self) -> tuple[int, ...]:
        return (self.dim_codomain,)

    @property
    @abstractmethod
    def domain_range(self) -> DomainRange:
        """Return the :term:`domain` range of the object

        Returns:
            List of tuples with the ranges for each domain dimension.
        """
        pass

    @override
    @property
    def domain(self) -> Region[InputDType]:
        lower = np.array([d[0] for d in self.domain_range])
        upper = np.array([d[1] for d in self.domain_range])

        return AxisAlignedBox(lower, upper)

    @abstractmethod
    def derivative(self, *, order: int = 1) -> Self:
        """
        Differentiate a FData object.

        Args:
            order: Order of the derivative. Defaults to one.

        Returns:
            Functional object containg the derivative.

        """
        pass

    @abstractmethod
    def integrate(
        self,
        *,
        domain: Optional[DomainRange] = None,
    ) -> NDArrayFloat:
        """
        Integration of the FData object.

        The integration is performed over the whole domain. Thus, for a
        function of several variables this will be a multiple integral.

        For a vector valued function the vector of integrals will be
        returned.

        Args:
            domain: Domain range where we want to integrate.
                By default is None as we integrate on the whole domain.

        Returns:
            NumPy array of size (``n_samples``, ``dim_codomain``)
            with the integrated data.

        """
        pass

    @abstractmethod
    def shift(
        self,
        shifts: Union[ArrayLike, float],
        *,
        restrict_domain: bool = False,
        extrapolation: AcceptedExtrapolation = "default",
        grid_points: Optional[GridPointsLike] = None,
    ) -> FDataGrid:
        r"""
        Perform a shift of the curves.

        The i-th shifted function :math:`y_i` has the form

        .. math::
            y_i(t) = x_i(t + \delta_i)

        where :math:`x_i` is the i-th original function and :math:`delta_i` is
        the shift performed for that function, that must be a vector in the
        domain space.

        Note that a positive shift moves the graph of the function in the
        negative direction and vice versa.

        Args:
            shifts: List with the shifts
                corresponding for each sample or numeric with the shift to
                apply to all samples.
            restrict_domain: If True restricts the domain to avoid the
                evaluation of points outside the domain using extrapolation.
                Defaults uses extrapolation.
            extrapolation: Controls the
                extrapolation mode for elements outside the domain range.
                By default uses the method defined in fd. See extrapolation to
                more information.
            grid_points: Grid of points where
                the functions are evaluated to obtain the discrete
                representation of the object to operate. If ``None`` the
                current grid_points are used to unificate the domain of the
                shifted data.

        Returns:
            Shifted functions.

        """
        assert grid_points is not None
        grid_points = check_grid_points(grid_points)

        arr_shifts = np.array([shifts] if np.isscalar(shifts) else shifts)

        # Accept unidimensional array when the domain dimension is one or when
        # the shift is the same for each sample
        if arr_shifts.ndim == 1:
            arr_shifts = (
                arr_shifts[np.newaxis, :]  # Same shift for each sample
                if len(arr_shifts) == self.dim_domain
                else arr_shifts[:, np.newaxis]
            )

        if len(arr_shifts) not in {1, self.n_samples}:
            raise ValueError(
                f"The length of the shift vector ({len(arr_shifts)}) must "
                f"have length equal to 1 or to the number of samples "
                f"({self.n_samples})",
            )

        if restrict_domain:
            domain = np.asarray(self.domain_range)

            a = domain[:, 0] - np.min(np.min(arr_shifts, axis=0), 0)
            b = domain[:, 1] - np.max(np.max(arr_shifts, axis=1), 0)

            domain = np.hstack((a, b))
            domain_range = tuple(domain)

        if len(arr_shifts) == 1:
            shifted_grid_points = tuple(
                g + s for g, s in zip(grid_points, arr_shifts[0])
            )
            data_matrix = self(
                shifted_grid_points,
                extrapolation=extrapolation,
                aligned=True,
                grid=True,
            )
        else:
            shifted_grid_points_per_sample = grid_points + arr_shifts
            data_matrix = self(
                shifted_grid_points_per_sample,
                extrapolation=extrapolation,
                aligned=False,
                grid=True,
            )

        shifted = self.to_grid().copy(
            data_matrix=data_matrix,
            grid_points=grid_points,
        )

        if restrict_domain:
            shifted = shifted.restrict(domain_range)

        return shifted
