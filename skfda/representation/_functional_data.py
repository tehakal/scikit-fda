"""Module for functional data manipulation.

Defines the abstract class that should be implemented by the funtional data
objects of the package and contains some commons methods.
"""

from __future__ import annotations

from abc import ABC

import pandas

from .._utils.ndfunction import NDFunction, concatenate as concatenate


class FData(  # noqa: WPS214
    ABC,
    NDFunction,
    pandas.api.extensions.ExtensionArray,  # type: ignore[misc]
):
    """
    Defines the structure of a functional data object.

    Attributes:
        n_samples (int): Number of samples.
        dim_domain (int): Dimension of the domain.
        dim_codomain (int): Dimension of the image.
        extrapolation (Extrapolation): Default extrapolation mode.
        dataset_name (str): name of the dataset.
        argument_names (tuple): tuple containing the names of the different
            arguments.
        coordinate_names (tuple): tuple containing the names of the different
            coordinate functions.

    """
