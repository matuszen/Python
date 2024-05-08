import numpy as np
from typing import Union
from collections.abc import Iterable


def euclidean_distance(
    first_point: Iterable[Union[int, float]],
    second_point: Iterable[Union[int, float]],
) -> float:
    """
    Calculate the Euclidean distance between two points.

    This function computes the Euclidean distance between two points in Euclidean space.
    The points are represented as iterables containing numerical values.

    Parameters
    ----------
    first_point : Iterable[int | float]
        The first point as an iterable sequence of numerical values.
    second_point : Iterable[int | float]
        The second point as an iterable sequence of numerical values.

    Returns
    -------
    float
        The Euclidean distance between the two points as a single float value.

    Raises
    ------
    ValueError
        If the input points are not of the same length.

    Examples
    --------
    >>> euclidean_distance((0, 0), (3, 4))
    5.0
    >>> euclidean_distance([1, 2, 3], [4, 5, 6])
    5.196152422706632
    >>> euclidean_distance((1.5, 2.5, 3.5), (4.5, 5.5, 6.5))
    5.196152422706632

    Notes
    -----
    This function assumes that both input points are of the same length and contain
    numerical values (either int or float).
    """

    if len(first_point) != len(second_point):
        raise ValueError("Input vectors must be of the same length.")

    return sum((v1 - v2) ** 2 for v1, v2 in zip(first_point, second_point)) ** (1 / 2)


def euclidean_distance_partialy_np(
    first_point: Iterable[Union[int, float]],
    second_point: Iterable[Union[int, float]],
) -> np.float64:
    """
    Calculate the Euclidean distance between two points using numpy for conversion.

    This function computes the Euclidean distance between two points in Euclidean space.
    The points are represented as iterables containing numerical values. The input
    iterables are converted to numpy arrays for validation purposes, but the computation
    is performed using standard Python operations.

    Parameters
    ----------
    first_point : Iterable[int | float]
        The first point as an iterable sequence of numerical values.
    second_point : Iterable[int | float]
        The second point as an iterable sequence of numerical values.

    Returns
    -------
    np.float64
        The Euclidean distance between the two points as a single numpy float64 value.

    Raises
    ------
    ValueError
        If the input points are not of the same length.

    Examples
    --------
    >>> euclidean_distance_partialy_np((0, 0), (3, 4))
    5.0
    >>> euclidean_distance_partialy_np([1, 2, 3], [4, 5, 6])
    5.196152422706632
    >>> euclidean_distance_partialy_np((1.5, 2.5, 3.5), (4.5, 5.5, 6.5))
    5.196152422706632

    Notes
    -----
    This function assumes that both input points are of the same length and contain
    numerical values (either int or float).
    """

    first_array: np.ndarray = np.fromiter(first_point, dtype=float)
    second_array: np.ndarray = np.fromiter(second_point, dtype=float)

    if first_array.shape[0] != second_array.shape[0]:
        raise ValueError("Input vectors must be of the same length.")

    return sum((v1 - v2) ** 2 for v1, v2 in zip(first_array, second_array)) ** (1 / 2)


def euclidean_distance_with_np(
    first_point: Iterable[Union[int, float]],
    second_point: Iterable[Union[int, float]],
) -> np.float64:
    """
    Calculate the Euclidean distance between two points using numpy.

    This function computes the Euclidean distance between two points in Euclidean space.
    The points are represented as iterables containing numerical values. The input
    iterables are converted to numpy arrays, and the computation is performed using
    numpy's vectorized operations for efficiency.

    Parameters
    ----------
    first_point : Iterable[int | float]
        The first point as an iterable sequence of numerical values.
    second_point : Iterable[int | float]
        The second point as an iterable sequence of numerical values.

    Returns
    -------
    np.float64
        The Euclidean distance between the two points as a single numpy float64 value.

    Raises
    ------
    ValueError
        If the input points are not of the same length.

    Examples
    --------
    >>> euclidean_distance_with_np((0, 0), (3, 4))
    5.0
    >>> euclidean_distance_with_np([1, 2, 3], [4, 5, 6])
    5.196152422706632
    >>> euclidean_distance_with_np((1.5, 2.5, 3.5), (4.5, 5.5, 6.5))
    5.196152422706632

    Notes
    -----
    This function assumes that both input points are of the same length and contain
    numerical values (either int or float).
    """

    if len(first_point) != len(second_point):
        raise ValueError("Input vectors must be of the same length.")

    return np.sqrt(np.sum((np.asarray(first_point) - np.asarray(second_point)) ** 2))


if __name__ == "__main__":
    import timeit

    first_list = [
        81286211.1234124124214,
        8638232.12412432324234,
        3124342235.124213142154124,
    ]
    second_list = [
        4796584.123154124,
        8921487215.3252141235412,
        6532431223.2131243253242,
    ]

    first_array = np.array(first_list)
    second_array = np.array(second_list)

    print(
        "Numpy functions with conversion to np.ndarray\n",
        "Python list as input:",
        timeit.timeit(
            "euclidean_distance_with_np(first_list, second_list)",
            number=10000,
            globals=globals(),
        ),
        "\n np.ndarray as input:",
        timeit.timeit(
            "euclidean_distance_with_np(first_array, second_array)",
            number=10000,
            globals=globals(),
        ),
    )

    print(
        "Python functions with conversion to np.ndarray\n",
        "Python list as input:",
        timeit.timeit(
            "euclidean_distance_partialy_np(first_list, second_list)",
            number=10000,
            globals=globals(),
        ),
        "\n np.ndarray as input:",
        timeit.timeit(
            "euclidean_distance_partialy_np(first_array, second_array)",
            number=10000,
            globals=globals(),
        ),
    )

    print(
        "Without Numpy\n",
        "Python list as input:",
        timeit.timeit(
            "euclidean_distance(first_list, second_list)",
            number=10000,
            globals=globals(),
        ),
        "\n np.ndarray as input:",
        timeit.timeit(
            "euclidean_distance(first_array, second_array)",
            number=10000,
            globals=globals(),
        ),
    )
