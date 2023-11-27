"""Routines for input validation and conversion."""
from __future__ import annotations

from typing import Sequence, TypeVar, cast

import numpy as np

from .._array_api import (
    Array,
    DType,
    Shape,
    array_namespace,
    is_array_api_obj,
    is_nested_array,
)
from ..typing import GridPoints, GridPointsLike

A = TypeVar('A', bound=Array[Shape, DType])


def check_grid_points(grid_points_like: GridPointsLike[A]) -> GridPoints[A]:
    """
    Convert to grid points.

    If the original list is one-dimensional (e.g. [1, 2, 3]), return list to
    array (in this case [array([1, 2, 3])]).

    If the original list is two-dimensional (e.g. [[1, 2, 3], [4, 5]]), return
    a list containing other one-dimensional arrays (in this case
    [array([1, 2, 3]), array([4, 5])]).

    In any other case the behaviour is unespecified.

    """
    if is_array_api_obj(grid_points_like):
        if is_nested_array(grid_points_like):
            return grid_points_like

        # It is an array
        grid_points = np.empty(shape=1, dtype=np.object_)
        grid_points[0] = grid_points_like
        return np.squeeze(grid_points)

    # It is a sequence!
    # Ensure that elements are compatible arrays

    # This cast won't be needed once PEP 724 is accepted
    grid_points_like = cast(Sequence[A], grid_points_like)

    array_namespace(*grid_points_like)
    grid_points = np.empty(shape=len(grid_points_like), dtype=np.object_)
    grid_points[...] = grid_points_like
    return grid_points
