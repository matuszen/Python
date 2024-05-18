import math
from collections.abc import Callable, Iterable

import numpy as np


def runge_kutta_method(
    x0: float,
    y0: float,
    h: float,
    x_end: float,
    fun: Callable[[float, float], float],
) -> Iterable[float]:
    """
    Solves an ordinary differential equation using the 4th-order Runge-Kutta method.

    Parameters
    ----------
    x0 : float
        The initial value of x.
    y0 : float
        The initial value of y.
    h : float
        The step size for the numerical solution.
    x_end : float
        The value of x at which to stop the computation.
    fun : (float, float) -> float
        A function that returns the derivative of y at given values of x and y.

    Returns
    -------
    Iterable[float]
        Array of y values at each step from x0 to x_end.

    Raises
    ------
    ValueError
        If `x0` is greater than or equal to `x_end`.
        If `h` is not positive.

    Examples
    --------
    >>> def fun(x, y):
    ...     return x + y
    >>> x0 = 0
    >>> y0 = 1
    >>> h = 0.1
    >>> x_end = 0.5
    >>> runge_kutta_method(x0, y0, h, x_end, fun)
    array([1.        , 1.11034167, 1.24280596, 1.39846959, 1.57852783, 1.78429709])

    Notes
    -----
    The 4th-order Runge-Kutta method is an iterative technique that provides a
    numerical solution to ordinary differential equations of the form dy/dx = f(x, y).
    The method is based on taking the average of four increments to estimate the
    value of y at the next step.
    """

    if x0 >= x_end:
        raise ValueError(
            "The final value of x must be greater than the first value of x"
        )

    if h <= 0:
        raise ValueError("Step size must be positive")

    n = int(math.ceil((x_end - x0) / h))
    y = np.zeros(n + 1)

    x = x0
    y[0] = y0
    sixth_h = h / 6

    for k in range(n):
        k1 = fun(x, y[k])
        h_k1_2 = 0.5 * h * k1
        k2 = fun(x + 0.5 * h, y[k] + h_k1_2)
        k3 = fun(x + 0.5 * h, y[k] + 0.5 * h * k2)
        k4 = fun(x + h, y[k] + h * k3)
        y[k + 1] = y[k] + sixth_h * (k1 + 2 * k2 + 2 * k3 + k4)
        x += h

    return y


if __name__ == "__main__":
    import doctest
    import timeit

    import matplotlib.pyplot as plt

    doctest.testmod()

    def fun(x: float, y: float) -> float:
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
            "runge_kutta_method(X0, Y0, H, X_END, fun)",
            number=1000,
            globals=globals(),
        ),
    )

    X = [round(X0 + H * i, 2) for i in range(int((X_END - X0) / H) + 1)]
    Y = runge_kutta_method(X0, Y0, H, X_END, fun)

    with plt.style.context("bmh"):
        fig, ax = plt.subplots()

        ax.plot(X, Y, "r", label="f(x, y) = x + 0.6 * y")

        ax.set_title("Runge-Kutta method")
        ax.set_xlabel("X Axis")
        ax.set_ylabel("Y Axis")

        fig.set_facecolor("#ddd")
        ax.set_facecolor("#ddd")

        plt.legend()
        plt.show()
