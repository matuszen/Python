from collections.abc import Callable

import numpy as np


def euler_method(
    f: Callable[[float, float], float],
    x0: float,
    y0: float,
    h: float,
    x_end: float,
) -> np.ndarray[tuple[int], float]:
    """
    Solves an ordinary differential equation using the Euler method.

    Parameters
    ----------
    f : (float, float) -> float
        A function that returns the derivative of y at given values of x and y.
    x0 : float
        The initial value of x.
    y0 : float
        The initial value of y.
    h : float
        The step size for the numerical solution.
    x_end : float
        The value of x at which to stop the computation.

    Returns
    -------
    ndarray[(int), float]
        One-dimensional array of y values at each step from x0 to x_end.

    Raises
    ------
    ValueError
        If `h` is not positive.
        If `x_end` is less than or equal to `x0`.

    Examples
    --------
    >>> def f(x, y):
    ...     return x + y
    >>> x0 = 0
    >>> y0 = 1
    >>> h = 0.1
    >>> x_end = 0.5
    >>> euler_method(f, x0, y0, h, x_end)
    array([1.     , 1.1    , 1.22   , 1.362  , 1.5282 , 1.72102])

    Notes
    -----
    The Euler method is a simple numerical procedure for solving ordinary differential
    equations of the form dy/dx = f(x, y). It is the most basic explicit method for
    numerical integration of ordinary differential equations and is the simplest
    Runge-Kutta method. The method involves taking steps of size `h` and approximating
    the slope at each step to compute the next value of y.
    """

    if h <= 0:
        raise ValueError("Step size must be positive")

    if x_end <= x0:
        raise ValueError(
            "The final value of x must be greater than the first value of x"
        )

    n = int((x_end - x0) / h)
    y = np.zeros(n + 1)

    x = x0
    y[0] = y0

    for i in range(n):
        y[i + 1] = y[i] + h * f(x, y[i])
        x += h

    return y


if __name__ == "__main__":
    import doctest
    import timeit

    import matplotlib.pyplot as plt

    doctest.testmod()

    def f(x: float, y: float) -> float:
        return x + 0.6 * y

    X0 = 0.0
    Y0 = 0.2
    # If H is smaller, the chart will be more accurate,
    # compare this chart with chart with H = 0.2
    H = 0.5
    X_END = 5

    print(
        "Execution time:",
        timeit.timeit(
            "euler_method(f, X0, Y0, H, X_END)",
            number=1000,
            globals=globals(),
        ),
    )

    X = [round(X0 + H * i, 2) for i in range(int((X_END - X0) / H) + 1)]
    Y = euler_method(f, X0, Y0, H, X_END)

    with plt.style.context("bmh"):
        fig, ax = plt.subplots()

        ax.plot(X, Y, "r", label="f(x, y) = x + 0.6 * y")

        ax.set_title("Euler method")
        ax.set_xlabel("X Axis")
        ax.set_ylabel("Y Axis")

        fig.set_facecolor("#ddd")
        ax.set_facecolor("#ddd")

        plt.legend()
        plt.show()
