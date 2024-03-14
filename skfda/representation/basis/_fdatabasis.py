from __future__ import annotations

import copy
import warnings
from builtins import isinstance
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Literal,
    Optional,
    Sequence,
    Type,
    TypeVar,
    Union,
    cast,
    overload,
)

import numpy as np
import pandas.api.extensions

from ..._utils import _check_array_key, _int_to_real, constants, nquad_vec
from ..._utils.ndfunction.extrapolation import ExtrapolationLike
from ..._utils.ndfunction.utils._points import input_points_batch_shape
from ...typing._base import DomainRange, GridPointsLike, LabelTupleLike
from ...typing._numpy import ArrayLike, NDArrayBool, NDArrayFloat, NDArrayInt
from .. import grid
from .._functional_data import FData

if TYPE_CHECKING:
    from .. import FDataGrid
    from . import Basis

T = TypeVar('T', bound='FDataBasis')


AcceptedExtrapolation = Union[ExtrapolationLike, None, Literal["default"]]


class FDataBasis(FData):  # noqa: WPS214
    r"""Basis representation of functional data.

    Class representation for functional data in the form of a set of basis
    functions multplied by a set of coefficients.

    .. math::
        f(x) = \sum_{k=1}{K}c_k\phi_k

    Where n is the number of basis functions, :math:`c = (c_1, c_2, ...,
    c_K)` the vector of coefficients and  :math:`\phi = (\phi_1, \phi_2,
    ..., \phi_K)` the basis function system.

    Attributes:
        basis: Basis function system.
        coefficients: List or matrix of coefficients. Has to
            have the same length or number of columns as the number of basis
            function in the basis. If a matrix, each row contains the
            coefficients that multiplied by the basis functions produce each
            functional datum.
        domain_range: 2 dimension matrix where each row
            contains the bounds of the interval in which the functional data
            is considered to exist for each one of the axes.
        dataset_name: name of the dataset.
        argument_names: tuple containing the names of the different
            arguments.
        coordinate_names: tuple containing the names of the different
            coordinate functions.
        extrapolation: defines the default type of
            extrapolation. By default None, which does not apply any type of
            extrapolation. See `Extrapolation` for detailled information of the
            types of extrapolation.

    Examples:
        >>> from skfda.representation.basis import FDataBasis, MonomialBasis
        >>>
        >>> basis = MonomialBasis(n_basis=4)
        >>> coefficients = [1, 1, 3, .5]
        >>> FDataBasis(basis, coefficients)
        FDataBasis(
            basis=MonomialBasis(domain_range=((0.0, 1.0),), n_basis=4),
            coefficients=[[ 1.   1.   3.   0.5]],
            ...)

    """

    def __init__(
        self,
        basis: Basis,
        coefficients: ArrayLike,
        *,
        dataset_name: Optional[str] = None,
        argument_names: Optional[LabelTupleLike] = None,
        coordinate_names: Optional[LabelTupleLike] = None,
        sample_names: Optional[LabelTupleLike] = None,
        extrapolation: Optional[ExtrapolationLike] = None,
    ) -> None:
        """Construct a FDataBasis object."""
        coefficients = _int_to_real(np.atleast_2d(coefficients))
        if coefficients.shape[1] != basis.n_basis:
            raise ValueError(
                "The length or number of columns of coefficients "
                "has to be the same equal to the number of "
                "elements of the basis.",
            )
        self.basis = basis
        self.coefficients = coefficients

        super().__init__(
            extrapolation=extrapolation,
            dataset_name=dataset_name,
            argument_names=argument_names,
            coordinate_names=coordinate_names,
            sample_names=sample_names,
        )

    @classmethod
    def from_data(
        cls,
        data_matrix: Union[NDArrayFloat, NDArrayInt],
        *,
        basis: Basis,
        grid_points: Optional[GridPointsLike] = None,
        sample_points: Optional[GridPointsLike] = None,
        method: str = 'cholesky',
    ) -> FDataBasis:
        r"""Transform raw data to a smooth functional form.

        Takes functional data in a discrete form and makes an approximates it
        to the closest function that can be generated by the basis. This
        function does not attempt to smooth the original data. If smoothing
        is desired, it is better to use :class:`BasisSmoother`.

        The fit is made so as to reduce the sum of squared errors
        [RS05-5-2-5]_:

        .. math::

            SSE(c) = (y - \Phi c)' (y - \Phi c)

        where :math:`y` is the vector or matrix of observations, :math:`\Phi`
        the matrix whose columns are the basis functions evaluated at the
        sampling points and :math:`c` the coefficient vector or matrix to be
        estimated.

        By deriving the first formula we obtain the closed formed of the
        estimated coefficients matrix:

        .. math::

            \hat{c} = \left( \Phi' \Phi \right)^{-1} \Phi' y

        The solution of this matrix equation is done using the cholesky
        method for the resolution of a LS problem. If this method throughs a
        rounding error warning you may want to use the QR factorisation that
        is more numerically stable despite being more expensive to compute.
        [RS05-5-2-7]_

        Args:
            data_matrix: List or matrix containing the
                observations. If a matrix each row represents a single
                functional datum and the columns the different observations.
            grid_points: Values of the domain where the previous
                data were taken.
            basis: Basis used.
            method: Algorithm used for calculating the coefficients using
                the least squares method. The values admitted are 'cholesky'
                and 'qr' for Cholesky and QR factorisation methods
                respectively.
            sample_points: Old name for `grid_points`. New code should
                use `grid_points` instead.

                .. deprecated:: 0.5

        Returns:
            FDataBasis: Represention of the data in a functional form as
                product of coefficients by basis functions.

        Examples:
            >>> import numpy as np
            >>> t = np.linspace(0, 1, 5)
            >>> x = np.sin(2 * np.pi * t) + np.cos(2 * np.pi * t) + 2
            >>> x
            array([ 3.,  3.,  1.,  1.,  3.])

            >>> from skfda.representation.basis import FDataBasis, FourierBasis
            >>> basis = FourierBasis((0, 1), n_basis=3)
            >>> fd = FDataBasis.from_data(x, grid_points=t, basis=basis)
            >>> fd.coefficients.round(2)
            array([[ 2.  , 0.71, 0.71]])

        References:
            .. [RS05-5-2-5] Ramsay, J., Silverman, B. W. (2005). How spline
                smooths are computed. In *Functional Data Analysis*
                (pp. 86-87). Springer.

            .. [RS05-5-2-7] Ramsay, J., Silverman, B. W. (2005). HSpline
                smoothing as an augmented least squares problem. In *Functional
                Data Analysis* (pp. 86-87). Springer.

        """
        if sample_points is not None:
            warnings.warn(
                "Parameter sample_points is deprecated. Use the "
                "parameter grid_points instead.",
                DeprecationWarning,
            )
            grid_points = sample_points

        fd = grid.FDataGrid(data_matrix=data_matrix, grid_points=grid_points)

        return fd.to_basis(basis=basis, method=method)

    @property
    def n_samples(self) -> int:
        return len(self.coefficients)

    @property
    def dim_domain(self) -> int:
        return self.basis.dim_domain

    @property
    def dim_codomain(self) -> int:
        return self.basis.dim_codomain

    @property
    def coordinates(self: T) -> _CoordinateIterator[T]:
        r"""Return a component of the FDataBasis.

        If the functional object contains samples
        :math:`f: \mathbb{R}^n \rightarrow \mathbb{R}^d`, this object allows
        a component of the vector :math:`f = (f_1, ..., f_d)`.

        """
        return _CoordinateIterator(self)

    @property
    def n_basis(self) -> int:
        """Return number of basis."""
        return self.basis.n_basis

    @property
    def domain_range(self) -> DomainRange:

        return self.basis.domain_range

    def _evaluate(
        self,
        input_points: NDArrayFloat,
        *,
        aligned: bool = True,
    ) -> NDArrayFloat:

        batch_shape = input_points_batch_shape(
            input_points,
            function=self,
            aligned=aligned,
        )

        if aligned:

            # Each row contains the values of one element of the basis
            basis_values = self.basis(input_points)

            res = np.tensordot(self.coefficients, basis_values, axes=(1, 0))

            return res.reshape(self.shape + batch_shape + self.output_shape)

        res_list = [
            np.sum((c * self.basis(np.asarray(p)).T).T, axis=0)
            for c, p in zip(self.coefficients, input_points)
        ]

        return np.reshape(
            res_list,
            self.shape + batch_shape + self.output_shape,
        )

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
        grid_points = (
            self._default_grid_points() if grid_points is None
            else grid_points
        )

        return super().shift(
            shifts=shifts,
            restrict_domain=restrict_domain,
            extrapolation=extrapolation,
            grid_points=grid_points,
        )

    def derivative(self: T, *, order: int = 1) -> T:  # noqa: D102

        if order < 0:
            raise ValueError("order only takes non-negative integer values.")

        if order == 0:
            return self.copy()

        basis, coefficients = self.basis.derivative_basis_and_coefs(
            self.coefficients,
            order,
        )

        return self.copy(basis=basis, coefficients=coefficients)

    def integrate(
        self: T,
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

        Examples:
            We first create the data basis.
                >>> from skfda.representation.basis import FDataBasis
                >>> from skfda.representation.basis import MonomialBasis
                >>> basis = MonomialBasis(n_basis=4)
                >>> coefficients = [1, 1, 3, .5]
                >>> fdata = FDataBasis(basis, coefficients)

            Then we can integrate on the whole domain.
                >>> fdata.integrate()
                array([[ 2.625]])

            Or we can do it on a given domain.
                >>> fdata.integrate(domain=((0.5, 1),))
                array([[ 1.8671875]])

        """
        if domain is None:
            domain = self.basis.domain_range

        integrated = nquad_vec(
            self,
            domain,
        )

        return integrated[:, :]

    def sum(  # noqa: WPS125
        self: T,
        *,
        axis: Optional[int] = None,
        out: None = None,
        keepdims: bool = False,
        skipna: bool = False,
        min_count: int = 0,
    ) -> T:
        """Compute the sum of all the samples in a FDataBasis object.

        Args:
            axis: Used for compatibility with numpy. Must be None or 0.
            out: Used for compatibility with numpy. Must be None.
            keepdims: Used for compatibility with numpy. Must be False.
            skipna: Wether the NaNs are ignored or not.
            min_count: Number of valid (non NaN) data to have in order
                for the a variable to not be NaN when `skipna` is
                `True`.

        Returns:
            A FDataBais object with just one sample
            representing the sum of all the samples in the original
            FDataBasis object.

        Examples:
            >>> from skfda.representation.basis import (
            ...     FDataBasis,
            ...     MonomialBasis,
            ... )
            >>> basis = MonomialBasis(n_basis=4)
            >>> coefficients = [[0.5, 1, 2, .5], [1.5, 1, 4, .5]]
            >>> FDataBasis(basis, coefficients).sum()
            FDataBasis(
                basis=MonomialBasis(domain_range=((0.0, 1.0),), n_basis=4),
                coefficients=[[ 2.  2.  6.  1.]],
                ...)

        """
        super().sum(axis=axis, out=out, keepdims=keepdims, skipna=skipna)

        coefs = (
            np.nansum(self.coefficients, axis=0) if skipna
            else np.sum(self.coefficients, axis=0)
        )

        if min_count > 0:
            valid = ~np.isnan(self.coefficients)
            n_valid = np.sum(valid, axis=0)
            coefs[n_valid < min_count] = np.NaN

        return self.copy(
            coefficients=coefs,
            sample_names=(None,),
        )

    def var(
        self: T,
        eval_points: Optional[NDArrayFloat] = None,
        correction: int = 0,
    ) -> T:
        """Compute the variance of the functional data object.

        A numerical approach its used. The object its transformed into its
        discrete representation and then the variance is computed and
        then the object is taken back to the basis representation.

        Args:
            eval_points: Set of points where the
                functions are evaluated to obtain the discrete
                representation of the object. If none are passed it calls
                numpy.linspace with bounds equal to the ones defined in
                self.domain_range and the number of points the maximum
                between 501 and 10 times the number of basis.
            correction: degrees of freedom adjustment. The divisor used in the
                calculation is `N - correction`, where `N` represents the
                number of elements. Default: `0`.

        Returns:
            Variance of the original object.

        """
        return self.to_grid(
            eval_points,
        ).var(correction=correction).to_basis(self.basis)

    @overload
    def cov(  # noqa: WPS451
        self: T,
        s_points: NDArrayFloat,
        t_points: NDArrayFloat,
        /,
        correction: int = 0,
    ) -> NDArrayFloat:
        pass

    @overload
    def cov(    # noqa: WPS451
        self: T,
        /,
        correction: int = 0,
    ) -> Callable[[NDArrayFloat, NDArrayFloat], NDArrayFloat]:
        pass

    def cov(  # noqa: WPS320, WPS451
        self: T,
        s_points: Optional[NDArrayFloat] = None,
        t_points: Optional[NDArrayFloat] = None,
        /,
        correction: int = 0,
    ) -> Union[
        Callable[[NDArrayFloat, NDArrayFloat], NDArrayFloat],
        NDArrayFloat,
    ]:
        """Compute the covariance of the functional data object.

        Calculates the unbiased sample covariance function of the data.
        This is expected to be only defined for univariate functions.
        This is a function defined over the basis consisting of the tensor
        product of the original basis with itself. The resulting covariance
        function is then represented as a callable object.

        If s_points or t_points are not provided, this method returns
        a callable object representing the covariance function.
        If s_points and t_points are provided, this method returns the
        evaluation of the covariance function at the grid formed by the
        cartesian product of the points in s_points and t_points.

        Args:
            s_points: Points where the covariance function is evaluated.
            t_points: Points where the covariance function is evaluated.
            correction: degrees of freedom adjustment. The divisor used in the
                calculation is `N - correction`, where `N` represents the
                number of elements. Default: `0`.

        Returns:
            Covariance function.

        """
        # To avoid circular imports
        from ...misc.covariances import EmpiricalBasis
        cov_function = EmpiricalBasis(self, correction=correction)
        if s_points is None or t_points is None:
            return cov_function
        return cov_function(s_points, t_points)

    def to_grid(
        self,
        grid_points: Optional[GridPointsLike] = None,
        *,
        sample_points: Optional[GridPointsLike] = None,
    ) -> FDataGrid:
        """Return the discrete representation of the object.

        Args:
            grid_points (array_like, optional): Points per axis where the
                functions are evaluated. If none are passed it calls
                numpy.linspace with bounds equal to the ones defined in
                self.domain_range and the number of points the maximum
                between 501 and 10 times the number of basis.
            sample_points: Old name for `grid_points`. New code should
                use `grid_points` instead.

                .. deprecated:: 0.5

        Returns:
              FDataGrid: Discrete representation of the functional data
              object.

        Examples:
            >>> from skfda.representation.basis import(
            ...     FDataBasis,
            ...     MonomialBasis,
            ... )
            >>> fd = FDataBasis(
            ...     coefficients=[[1, 1, 1], [1, 0, 1]],
            ...     basis=MonomialBasis(domain_range=(0,5), n_basis=3),
            ... )
            >>> fd.to_grid(np.array([0, 1, 2]))
            FDataGrid(
                array([[[ 1.],
                        [ 3.],
                        [ 7.]],
                       [[ 1.],
                        [ 2.],
                        [ 5.]]]),
                grid_points=array([array([0, 1, 2])], dtype=object),
                domain_range=((0.0, 5.0),),
                ...)

        """
        if sample_points is not None:
            warnings.warn(
                "Parameter sample_points is deprecated. Use the "
                "parameter grid_points instead.",
                DeprecationWarning,
            )
            grid_points = sample_points

        if grid_points is None:
            grid_points = self._default_grid_points()

        return grid.FDataGrid(
            self(grid_points, grid=True),
            grid_points=grid_points,
            domain_range=self.domain_range,
        )

    def to_basis(
        self,
        basis: Optional[Basis] = None,
        eval_points: Optional[NDArrayFloat] = None,
        **kwargs: Any,
    ) -> FDataBasis:
        """
        Return the basis representation of the object.

        Args:
            basis: Basis object in which the functional data are
                going to be represented.
            eval_points: Evaluation points used to discretize the function
                if the basis is going to be changed.
            kwargs: Keyword arguments to be passed to
                FDataBasis.from_data().

        Returns:
            Basis representation of the funtional data object.

        """
        if basis is None or basis == self.basis:
            return self.copy()

        return self.to_grid(grid_points=eval_points).to_basis(basis, **kwargs)

    def copy(
        self: T,
        *,
        deep: bool = False,  # For Pandas compatibility
        basis: Optional[Basis] = None,
        coefficients: Optional[NDArrayFloat] = None,
        dataset_name: Optional[str] = None,
        argument_names: Optional[LabelTupleLike] = None,
        coordinate_names: Optional[LabelTupleLike] = None,
        sample_names: Optional[LabelTupleLike] = None,
        extrapolation: Optional[ExtrapolationLike] = None,
    ) -> T:
        """Copy the FDataBasis."""
        if basis is None:
            basis = copy.deepcopy(self.basis)

        if coefficients is None:
            coefficients = self.coefficients

        if dataset_name is None:
            dataset_name = self.dataset_name

        if argument_names is None:
            # Tuple, immutable
            argument_names = self.argument_names

        if coordinate_names is None:
            # Tuple, immutable
            coordinate_names = self.coordinate_names

        if sample_names is None:
            # Tuple, immutable
            sample_names = self.sample_names

        if extrapolation is None:
            extrapolation = self.extrapolation

        return FDataBasis(
            basis,
            coefficients,
            dataset_name=dataset_name,
            argument_names=argument_names,
            coordinate_names=coordinate_names,
            sample_names=sample_names,
            extrapolation=extrapolation,
        )

    def _default_grid_points(self) -> GridPointsLike:
        npoints = constants.N_POINTS_FINE_MESH
        return [
            np.linspace(*r, npoints)
            for r in self.domain_range
        ]

    def _to_R(self) -> str:  # noqa: N802
        """Return the code to build the object on fda package on R."""
        return (
            f"fd("  # noqa: WPS437
            f"coef = {self._array_to_R(self.coefficients, transpose=True)},"
            f" basisobj = {self.basis._to_R()})"
        )

    def _array_to_R(  # noqa: N802
        self,
        coefficients: NDArrayFloat,
        transpose: bool = False,
    ) -> str:
        if coefficients.ndim == 1:
            coefficients = coefficients[None]

        if transpose is True:
            coefficients = np.transpose(coefficients)

        (rows, cols) = coefficients.shape
        retstring = "matrix(c("
        retstring += "".join(
            f"{coefficients[i, j]}, "
            for j in range(cols)
            for i in range(rows)
        )

        return (
            retstring[:len(retstring) - 2]
            + f"), nrow = {rows}, ncol = {cols})"
        )

    def __repr__(self) -> str:

        return (
            f"{self.__class__.__name__}("  # noqa: WPS221
            f"\nbasis={self.basis},"
            f"\ncoefficients={self.coefficients},"
            f"\ndataset_name={self.dataset_name},"
            f"\nargument_names={repr(self.argument_names)},"
            f"\ncoordinate_names={repr(self.coordinate_names)},"
            f"\nextrapolation={self.extrapolation})"
        ).replace('\n', '\n    ')

    def __str__(self) -> str:

        return (
            f"{self.__class__.__name__}("
            f"\n_basis={self.basis},"
            f"\ncoefficients={self.coefficients})"
        ).replace('\n', '\n    ')

    def equals(self, other: object) -> bool:
        """Equality of FDataBasis."""
        # TODO check all other params

        if not super().equals(other):
            return False

        other = cast(grid.FDataGrid, other)

        return (
            self.basis == other.basis
            and np.array_equal(self.coefficients, other.coefficients)
        )

    def _eq_elemenwise(self: T, other: T) -> NDArrayBool:
        """Elementwise equality of FDataBasis."""
        return np.all(  # type: ignore[no-any-return]
            self.coefficients == other.coefficients,
            axis=1,
        )

    def concatenate(
        self: T,
        *others: T,
        as_coordinates: bool = False,
    ) -> T:
        """
        Join samples from a similar FDataBasis object.

        Joins samples from another FDataBasis object if they have the same
        basis.

        Args:
            others: Objects to be concatenated.
            as_coordinates:  If False concatenates as
                new samples, else, concatenates the other functions as
                new components of the image. Defaults to False.

        Returns:
            :class:`FDataBasis`: FDataBasis object with the samples from the
            original objects.

        Todo:
            By the moment, only unidimensional objects are supported in basis
            representation.

        """
        # TODO: Change to support multivariate functions
        #  in basis representation
        if as_coordinates:
            return NotImplemented

        for other in others:
            if other.basis != self.basis:
                raise ValueError("The objects should have the same basis.")

        data = [self.coefficients] + [other.coefficients for other in others]

        sample_names = [fd.sample_names for fd in (self, *others)]

        return self.copy(
            coefficients=np.concatenate(data, axis=0),
            sample_names=sum(sample_names, ()),
        )

    def compose(
        self,
        fd: FData,
        *,
        eval_points: Optional[NDArrayFloat] = None,
        **kwargs: Any,
    ) -> FData:
        """
        Composition of functions.

        Performs the composition of functions. The basis is discretized to
        compute the composition.

        Args:
            fd: FData object to make the composition. Should
                have the same number of samples and image dimension equal to 1.
            eval_points: Points to perform the evaluation.
             kwargs: Named arguments to be passed to :func:`from_data`.

        Returns:
            Function resulting from the composition.

        """
        fd_grid = self.to_grid().compose(fd, eval_points=eval_points)

        if fd.dim_domain == 1:
            basis = self.basis.rescale(fd.domain_range[0])
            composition = fd_grid.to_basis(basis, **kwargs)
        else:
            #  Cant be convertered to basis due to the dimensions
            composition = fd_grid

        return composition

    def __getitem__(
        self: T,
        key: Union[int, slice, NDArrayInt, NDArrayBool],
    ) -> T:
        """Return self[key]."""
        key = _check_array_key(self.coefficients, key)

        return self.copy(
            coefficients=self.coefficients[key],
            sample_names=list(np.array(self.sample_names)[key]),
        )

    def __add__(
        self: T,
        other: T,
    ) -> T:
        """Addition for FDataBasis object."""
        if isinstance(other, FDataBasis) and self.basis == other.basis:

            return self._copy_op(
                other,
                basis=self.basis,
                coefficients=self.coefficients + other.coefficients,
            )

        return NotImplemented

    def __radd__(
        self: T,
        other: T,
    ) -> T:
        """Addition for FDataBasis object."""
        if isinstance(other, FDataBasis) and self.basis == other.basis:

            return self._copy_op(
                other,
                basis=self.basis,
                coefficients=self.coefficients + other.coefficients,
            )

        return NotImplemented

    def __sub__(
        self: T,
        other: T,
    ) -> T:
        """Subtraction for FDataBasis object."""
        if isinstance(other, FDataBasis) and self.basis == other.basis:

            return self._copy_op(
                other,
                basis=self.basis,
                coefficients=self.coefficients - other.coefficients,
            )

        return NotImplemented

    def __rsub__(
        self: T,
        other: T,
    ) -> T:
        """Right subtraction for FDataBasis object."""
        if isinstance(other, FDataBasis) and self.basis == other.basis:

            return self._copy_op(
                other,
                basis=self.basis,
                coefficients=other.coefficients - self.coefficients,
            )

        return NotImplemented

    def _mul_scalar(
        self: T,
        other: Union[NDArrayFloat, NDArrayInt, float],
    ) -> T:
        """Multiplication by scalar."""
        try:
            vector = _int_to_real(np.atleast_1d(other))
        except Exception:
            return NotImplemented

        if vector.ndim > 1:
            return NotImplemented

        vector = vector[:, np.newaxis]

        return self._copy_op(
            other,
            basis=self.basis,
            coefficients=self.coefficients * vector,
        )

    def __mul__(
        self: T,
        other: Union[NDArrayFloat, NDArrayInt, float],
    ) -> T:
        """Multiplication for FDataBasis object."""
        return self._mul_scalar(other)

    def __rmul__(
        self: T,
        other: Union[NDArrayFloat, NDArrayInt, float],
    ) -> T:
        """Multiplication for FDataBasis object."""
        return self._mul_scalar(other)

    def __truediv__(
        self: T,
        other: Union[NDArrayFloat, NDArrayInt, float],
    ) -> T:
        """Division for FDataBasis object."""
        try:
            other = 1 / np.asarray(other)
        except Exception:
            return NotImplemented

        return self._mul_scalar(other)

    def __rtruediv__(
        self: T,
        other: Union[NDArrayFloat, NDArrayInt, float],
    ) -> T:
        """Right division for FDataBasis object."""
        return NotImplemented

    def __neg__(self: T) -> T:
        """Negation of FData object."""
        return self.copy(coefficients=-self.coefficients)

    #####################################################################
    # Pandas ExtensionArray methods
    #####################################################################
    def _take_allow_fill(
        self: T,
        indices: NDArrayInt,
        fill_value: T,
    ) -> T:
        result = self.copy()
        result.coefficients = np.full(
            (len(indices),) + self.coefficients.shape[1:],
            np.nan,
        )

        positive_mask = indices >= 0
        result.coefficients[positive_mask] = self.coefficients[
            indices[positive_mask]
        ]

        if fill_value is not self.dtype.na_value:
            result.coefficients[~positive_mask] = fill_value.coefficients[0]

        return result

    @property
    def dtype(self) -> FDataBasisDType:
        """The dtype for this extension array, FDataGridDType"""
        return FDataBasisDType(basis=self.basis)

    @property
    def nbytes(self) -> int:
        """
        The number of bytes needed to store this object in memory.
        """
        return self.coefficients.nbytes

    def isna(self) -> NDArrayBool:
        """
        Return a 1-D array indicating if each value is missing.

        Returns:
            na_values (np.ndarray): Positions of NA.
        """
        return np.all(  # type: ignore[no-any-return]
            np.isnan(self.coefficients),
            axis=1,
        )


class FDataBasisDType(
    pandas.api.extensions.ExtensionDtype,  # type: ignore[misc]
):
    """DType corresponding to FDataBasis in Pandas."""

    kind = 'O'
    type = FDataBasis  # noqa: WPS125
    name = 'FDataBasis'
    na_value = pandas.NA

    _metadata = ("basis")

    def __init__(self, basis: Basis) -> None:
        self.basis = basis

    @classmethod
    def construct_array_type(cls) -> Type[FDataBasis]:  # noqa: D102
        return FDataBasis

    def _na_repr(self) -> FDataBasis:
        return FDataBasis(
            basis=self.basis,
            coefficients=((np.NaN,) * self.basis.n_basis,),
        )

    def __eq__(self, other: Any) -> bool:
        """
        Compare dtype equality.

        Rules for equality (similar to categorical):
        1) Any FData is equal to the string 'category'
        2) Any FData is equal to itself
        3) Otherwise, they are equal if the arguments are equal.
        6) Any other comparison returns False
        """
        if isinstance(other, str):
            return other == self.name
        elif other is self:
            return True

        return (
            isinstance(other, FDataBasisDType)
            and self.basis == other.basis
        )

    def __hash__(self) -> int:
        return hash(self.basis)


class _CoordinateIterator(Sequence[T]):
    """Internal class to iterate through the image coordinates.

    Dummy object. Should be change to support multidimensional objects.

    """

    def __init__(self, fdatabasis: T) -> None:
        """Create an iterator through the image coordinates."""
        self._fdatabasis = fdatabasis

    def __getitem__(self, key: Union[int, slice]) -> T:
        """Get a specific coordinate."""
        basis, coefs = self._fdatabasis.basis.coordinate_basis_and_coefs(
            self._fdatabasis.coefficients,
            key,
        )

        coord_names = self._fdatabasis.coordinate_names[key]
        if coord_names is None or isinstance(coord_names, str):
            coord_names = (coord_names,)

        return self._fdatabasis.copy(
            basis=basis,
            coefficients=coefs,
            coordinate_names=coord_names,
        )

    def __len__(self) -> int:
        """Return the number of coordinates."""
        return self._fdatabasis.dim_codomain
