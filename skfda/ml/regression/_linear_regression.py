from __future__ import annotations

import itertools
import warnings
from typing import Any, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted

from ...misc.lstsq import solve_regularized_weighted_lstsq
from ...misc.regularization import L2Regularization, compute_penalty_matrix
from ...representation import FData
from ...representation.basis import Basis
from ._coefficients import CoefficientInfo, coefficient_info_from_covariate

RegularizationType = Union[
    L2Regularization[Any],
    Sequence[Optional[L2Regularization[Any]]],
    None,
]

RegularizationIterableType = Union[
    L2Regularization[Any],
    Iterable[Optional[L2Regularization[Any]]],
    None,
]

AcceptedDataType = Union[
    FData,
    np.ndarray,
]

AcceptedDataCoefsType = Union[
    CoefficientInfo[FData],
    CoefficientInfo[np.ndarray],
]

BasisCoefsType = Sequence[Optional[Basis]]

ArgcheckResultType = Tuple[
    List[AcceptedDataType],
    np.ndarray,
    Optional[np.ndarray],
    List[AcceptedDataCoefsType],
]


class LinearRegression(
    BaseEstimator,  # type: ignore
    RegressorMixin,  # type: ignore
):
    r"""Linear regression with multivariate response.

    This is a regression algorithm equivalent to multivariate linear
    regression, but accepting also functional data expressed in a basis
    expansion.

    The model assumed by this method is:

    .. math::
        y = w_0 + w_1 x_1 + \ldots + w_p x_p + \int w_{p+1}(t) x_{p+1}(t) dt \
        + \ldots + \int w_r(t) x_r(t) dt

    where the covariates can be either multivariate or functional and the
    response is multivariate.

    .. warning::
        For now, only scalar responses are supported.

    Args:
        coef_basis (iterable): Basis of the coefficient functions of the
            functional covariates. If multivariate data is supplied, their
            corresponding entries should be ``None``. If ``None`` is provided
            for a functional covariate, the same basis is assumed. If this
            parameter is ``None`` (the default), it is assumed that ``None``
            is provided for all covariates.
        fit_intercept:  Whether to calculate the intercept for this
            model. If set to False, no intercept will be used in calculations
            (i.e. data is expected to be centered).
        regularization (int, iterable or :class:`Regularization`): If it is
            not a :class:`Regularization` object, linear differential
            operator regularization is assumed. If it
            is an integer, it indicates the order of the
            derivative used in the computing of the penalty matrix. For
            instance 2 means that the differential operator is
            :math:`f''(x)`. If it is an iterable, it consists on coefficients
            representing the differential operator used in the computing of
            the penalty matrix. For instance the tuple (1, 0,
            numpy.sin) means :math:`1 + sin(x)D^{2}`. If not supplied this
            defaults to 2. Only used if penalty_matrix is
            ``None``.

    Attributes:
        coef\_: A list containing the weight coefficient for each
            covariate. For multivariate data, the covariate is a Numpy array.
            For functional data, the covariate is a FDataBasis object.
        intercept\_: Independent term in the linear model. Set to 0.0
            if `fit_intercept = False`.

    Examples:
        >>> from skfda.ml.regression import LinearRegression
        >>> from skfda.representation.basis import (FDataBasis, Monomial,
        ...                                         Constant)

        Multivariate linear regression can be used with functions expressed in
        a basis. Also, a functional basis for the weights can be specified:

        >>> x_basis = Monomial(n_basis=3)
        >>> x_fd = FDataBasis(x_basis, [[0, 0, 1],
        ...                             [0, 1, 0],
        ...                             [0, 1, 1],
        ...                             [1, 0, 1]])
        >>> y = [2, 3, 4, 5]
        >>> linear = LinearRegression()
        >>> _ = linear.fit(x_fd, y)
        >>> linear.coef_[0]
        FDataBasis(
            basis=Monomial(domain_range=((0, 1),), n_basis=3),
            coefficients=[[-15.  96. -90.]],
            ...)
        >>> linear.intercept_
        array([ 1.])
        >>> linear.predict(x_fd)
        array([ 2.,  3.,  4.,  5.])

        Covariates can include also multivariate data:

        >>> x_basis = Monomial(n_basis=2)
        >>> x_fd = FDataBasis(x_basis, [[0, 2],
        ...                             [0, 4],
        ...                             [1, 0],
        ...                             [2, 0],
        ...                             [1, 2],
        ...                             [2, 2]])
        >>> x = [[1, 7], [2, 3], [4, 2], [1, 1], [3, 1], [2, 5]]
        >>> y = [11, 10, 12, 6, 10, 13]
        >>> linear = LinearRegression(
        ...              coef_basis=[None, Constant()])
        >>> _ = linear.fit([x, x_fd], y)
        >>> linear.coef_[0]
        array([ 2.,  1.])
        >>> linear.coef_[1]
        FDataBasis(
        basis=Constant(domain_range=((0, 1),), n_basis=1),
        coefficients=[[ 1.]],
        ...)
        >>> linear.intercept_
        array([ 1.])
        >>> linear.predict([x, x_fd])
        array([ 11.,  10.,  12.,   6.,  10.,  13.])

    """

    def __init__(
        self,
        *,
        coef_basis: Optional[BasisCoefsType] = None,
        fit_intercept: bool = True,
        regularization: RegularizationType = None,
        y_regularization: RegularizationType = None,
    ) -> None:
        self.coef_basis = coef_basis
        self.fit_intercept = fit_intercept
        self.regularization = regularization
        self.y_regularization = y_regularization

    def fit(  # noqa: D102
        self,
        X: Union[AcceptedDataType, Sequence[AcceptedDataType]],
        y: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
    ) -> LinearRegression:

        X_new, y, sample_weight, coef_info = self._argcheck_X_y(
            X,
            y,
            sample_weight,
            self.coef_basis,
        )

        regularization, y_regularization, lambdas = self._check_regularization(
            self.regularization, self.y_regularization, self.y_nbasis,
        )

        if self.fit_intercept:
            X_new, coef_info = self._concatenate_intercept(X_new, y, coef_info)

        penalty_matrix = compute_penalty_matrix(
            basis_iterable=(c.basis for c in coef_info),
            regularization_parameter=1,
            regularization=regularization,
        )

        y_penalty_matrix = compute_penalty_matrix(
            basis_iterable=(c.y_basis for c in coef_info),
            regularization_parameter=1,
            regularization=y_regularization,
        )

        if self.fit_intercept and penalty_matrix is not None:
            # Intercept is not penalized
            penalty_matrix[0, 0] = 0

        if self.functional_response:
            lambda_matrix = np.diag(lambdas)
            J_phi_theta = self.y_basis.inner_product_matrix(self.coef_basis[0])
            J_theta = self.coef_basis[0].inner_product_matrix()

            X_col_gram_mat = X_new.T @ X_new
            J_theta_kron_X_col_gram_mat = np.kron(J_theta, X_col_gram_mat)

            if penalty_matrix is not None:
                reg_matrix = np.kron(penalty_matrix, lambda_matrix)
                J_theta_kron_X_col_gram_mat += reg_matrix

            if y_penalty_matrix is not None:
                y_reg_matrix = np.kron(
                    y_penalty_matrix,
                    X_col_gram_mat,
                ) * self.y_lambda_parameter

                J_theta_kron_X_col_gram_mat += y_reg_matrix

            Xt_c_J_phi_theta = X_new.T @ y.coefficients @ J_phi_theta
            vec_Xt_c_J_phi_theta = np.reshape(Xt_c_J_phi_theta, (-1, 1), order='F')

            basiscoefs = np.linalg.solve(
                J_theta_kron_X_col_gram_mat,
                vec_Xt_c_J_phi_theta,
            )

            basiscoef_list = np.reshape(
                basiscoefs,
                (X_new.shape[1], -1),
                order='F',
            )
        else:
            inner_products_list = [
                c.regression_matrix(x, y)
                for x, c in zip(X_new, coef_info)
            ]
            # This is C @ J
            inner_products = np.concatenate(inner_products_list, axis=1)

            if sample_weight is not None:
                inner_products = inner_products * np.sqrt(sample_weight)
                y = y * np.sqrt(sample_weight)

            basiscoefs = solve_regularized_weighted_lstsq(
                coefs=inner_products,
                result=y,
                penalty_matrix=penalty_matrix,
            )

            coef_lengths = np.array([i.shape[1] for i in inner_products_list])
            coef_start = np.cumsum(coef_lengths)
            basiscoef_list = np.split(basiscoefs, coef_start)

        # Express the coefficients in functional form
        coefs = [
            c.convert_from_constant_coefs(bcoefs)
            for c, bcoefs in zip(coef_info, basiscoef_list)
        ]

        if self.fit_intercept:
            self.intercept_ = coefs[0]
            coefs = coefs[1:]
            self._coef_info_intercept = coef_info[0]
            coef_info = coef_info[1:]
        else:
            self.intercept_ = np.zeros(self.y_nbasis)

        self.coef_ = coefs
        self.basis_coefs = basiscoef_list
        self._coef_info = coef_info
        self._target_ndim = y.ndim

        return self

    def predict(  # noqa: D102
        self,
        X: Union[AcceptedDataType, Sequence[AcceptedDataType]],
    ) -> np.ndarray:

        check_is_fitted(self)
        X = self._argcheck_X(X)

        if self.functional_response:
            result_list = np.dot(X, self.basis_coefs)
            result = [
                coef_info.convert_from_constant_coefs(arr)
                for arr, coef_info
                in zip(result_list, self._coef_info)
            ]
        else:
            result = np.sum(
                [
                    coef_info.inner_product(coef, x)
                    for coef, x, coef_info
                    in zip(self.coef_, X, self._coef_info)
                ],
                axis=0,
            )
        
        if self.fit_intercept:
            result += self.intercept_

        if self._target_ndim == 1 and not self.functional_response:
            result = result.ravel()

        return result

    def _check_regularization(
        self,
        regularization: RegularizationType,
        y_regularization: RegularizationType,
        dimension: int,
    ):

        if isinstance(regularization, Iterable):
            lambdas = list(
                map(lambda reg: reg.regularization_parameter, regularization),
            )
        elif regularization is not None:
            lambda_parameter = regularization.regularization_parameter
            lambdas = [lambda_parameter] * dimension
        else:
            lambdas = [0] * dimension

        if self.fit_intercept:
            if self.functional_response:
                lambdas = [0] + lambdas
            else:
                if isinstance(regularization, Iterable):
                    regularization = itertools.chain([None], regularization)
                elif regularization is not None:
                    regularization = (None, regularization)

        if y_regularization is not None:
            self.y_lambda_parameter = y_regularization.regularization_parameter

        return regularization, y_regularization, lambdas

    def _concatenate_intercept(
        self,
        X: Sequence[AcceptedDataType],
        y: AcceptedDataType,
        coef_info: List[AcceptedDataCoefsType],
    ):
        new_x = np.ones((len(y), 1))
        if self.functional_response:
            X_new = np.insert(X, 0, new_x.T, axis=1)
        else:
            X_new = [new_x] + X

        c_info = [coefficient_info_from_covariate(new_x, y)] + coef_info

        return X_new, c_info

    def _argcheck_X(
        self,
        X: Union[AcceptedDataType, Sequence[AcceptedDataType]],
    ) -> Sequence[AcceptedDataType]:
        if isinstance(X, (FData, np.ndarray)):
            X = [X]

        X = [x if isinstance(x, FData) else np.asarray(x) for x in X]

        return X

    def _argcheck_X_y(
        self,
        X: Union[AcceptedDataType, Sequence[AcceptedDataType]],
        y: Union[AcceptedDataType, Sequence[AcceptedDataType]],
        sample_weight: Optional[np.ndarray] = None,
        coef_basis: Optional[BasisCoefsType] = None,
    ) -> ArgcheckResultType:
        """Do some checks to types and shapes."""
        # TODO: Add support for Dataframes

        new_X = self._argcheck_X(X)

        if isinstance(y, FData):
            if y.n_samples != len(new_X):
                raise ValueError(
                    "The number of samples on independent and "
                    "dependent variables should be the same",
                )
            self.functional_response = True
            new_X = np.asarray(new_X)
            self.y_nbasis = y.n_basis
            self.y_basis = y.basis
            if coef_basis is None:
                self.coef_basis = [y.basis]

            if not isinstance(self.y_basis, Basis):
                raise TypeError(
                    f"y basis must be a Basis object, not {type(self.y_basis)}",
                )
        else:
            if any(len(y) != len(x) for x in new_X):
                raise ValueError(
                    "The number of samples on independent and "
                    "dependent variables should be the same",
                )
            self.functional_response = False
            self.y_nbasis = 1
            y = np.asarray(y)

        if coef_basis is None:
            coef_basis = [None] * len(new_X)

        if len(coef_basis) != len(new_X):
            coef_basis = coef_basis * len(new_X)

        coef_info = [
            coefficient_info_from_covariate(x, y, basis=b)
            for x, b in zip(new_X, coef_basis)
        ]

        if sample_weight is not None:

            if len(sample_weight) != len(y):
                raise ValueError(
                    "The number of sample weights should be "
                    "equal to the number of samples.",
                )

            if np.any(np.array(sample_weight) < 0):
                raise ValueError(
                    "The sample weights should be non negative values",
                )

        return new_X, y, sample_weight, coef_info
