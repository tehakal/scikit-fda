"""Feature extraction union for dimensionality reduction."""
from __future__ import annotations

from typing import Any, Union

from numpy import ndarray
from pandas import DataFrame, concat
from sklearn.pipeline import FeatureUnion

from ....representation.basis import FDataBasis
from ....representation.grid import FDataGrid


class FdaFeatureUnion(FeatureUnion):
    """Concatenates results of multiple functional transformer objects.

    This estimator applies a list of transformer objects in parallel to the
    input data, then concatenates the results (They can be either FDataGrid
    and FDataBasis objects or multivariate data itself).This is useful to
    combine several feature extraction mechanisms into a single transformer.
    Parameters of the transformers may be set using its name and the parameter
    name separated by a '__'. A transformer may be replaced entirely by
    setting the parameter with its name to another transformer,
    or removed by setting to 'drop'.

    Parameters:
        transformer_list: list of tuple
            List of tuple containing `(str, transformer)`. The first element
            of the tuple is name affected to the transformer while the
            second element is a scikit-learn transformer instance.
            The transformer instance can also be `"drop"` for it to be
            ignored.
        n_jobs: int
            Number of jobs to run in parallel.
            ``None`` means 1 unless in a :obj:`joblib.parallel_backend`
            context.
            ``-1`` means using all processors.
            The default value is None
        transformer_weights: dict
            Multiplicative weights for features per transformer.
            Keys are transformer names, values the weights.
            Raises ValueError if key not present in ``transformer_list``.
        verbose: bool
            If True, the time elapsed while fitting each transformer will be
            printed as it is completed. By default the value is False
        np_array_output: bool
            indicates if the transformed data is requested to be a NumPy array
            output. By default the value is False.

    Examples:
    Firstly we will import the Berkeley Growth Study data set
    >>> from skfda.datasets import fetch_growth
    >>> X = fetch_growth(return_X_y=True)[0]

    Then we need to import the transformers we want to use
    >>> from skfda.preprocessing.dim_reduction.feature_extraction import FPCA
    >>> from skfda.representation import EvaluationTransformer

    Finally we import the union and apply fit and transform
    >>> from skfda.preprocessing.dim_reduction.feature_extraction
    ... import FdaFeatureUnion
    >>> union = FdaFeatureUnion([
    ...    ("Eval", EvaluationTransformer()),
    ...    ("fpca", FPCA()), ], np_array_output=True)
    >>> union.fit_transform(X)
    """

    def __init__(
        self,
        transformer_list,
        *,
        n_jobs=None,
        transformer_weights=None,
        verbose=False,
        np_array_output=False,
    ) -> None:
        self.np_array_output = np_array_output
        super().__init__(
            transformer_list,
            n_jobs=n_jobs,
            transformer_weights=transformer_weights,
            verbose=verbose,
        )

    def _hstack(self, Xs) -> Union[DataFrame, ndarray, Any]:

        if self.np_array_output:
            for i in Xs:
                if isinstance(i, FDataGrid or FDataBasis):
                    raise TypeError(
                        "There are transformed instances of FDataGrid or "
                        "FDataBasis that can't be concatenated on a NumPy "
                        "array.",
                    )
            return super()._hstack(Xs)

        if not isinstance(Xs[0], FDataGrid or FDataBasis):
            raise TypeError(
                "Transformed instance is not of type FDataGrid or"
                " FDataBasis. It is " + type(Xs[0]),
            )

        frames = [DataFrame({Xs[0].dataset_name.lower(): Xs[0]})]

        for j in Xs[1:]:
            if isinstance(j, FDataGrid or FDataBasis):
                frames.append(
                    DataFrame({j.dataset_name.lower(): j}),
                )
            else:
                raise TypeError(
                    "Transformed instance is not of type FDataGrid or"
                    " FDataBasis. It is " + type(j),
                )

        return concat(frames, axis=1)
