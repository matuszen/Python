import math
from collections.abc import Callable, Iterable

import matplotlib.pyplot as plt


def euler_method(
    x0: float,
    y0: float,
    step_size: float,
    x_end: float,
    fun: Callable,
) -> Iterable[float]:
    """
    Approximates the solution to an ordinary differential equation
    using the Euler method.

    Parameters
    ----------
    x0 : float
        The initial value of x.
    y0 : float
        The initial value of y.
    step_size : float
        The step size for the approximation.
    x_end : float
        The value of x at which to stop the approximation.
    fun : Callable
        A function that computes the derivative of y at (x, y).

    Returns
    -------
    Iterable[float]
        A list of y values approximating the solution
        to the differential equation at each step.

    Examples
    --------
    >>> def dy_dx(x, y):
    ...     return x + y
    >>> euler_method(0, 1, 0.1, 0.5, dy_dx)
    [1, 1.1, 1.21, 1.331, 1.4641, 1.61051]

    Notes
    -----
    This function uses the Euler method, which is a first-order numerical procedure for
    solving ordinary differential equations with a given initial value.
    The method approximates the solution by taking steps of size `step_size` and using
    the derivative function `fun` to estimate the slope of the solution at each step.
    """

    n = math.ceil((x_end - x0) / step_size)

    y = [0 for _ in range(n + 1)]
    y[0] = y0

    x = x0

    for i in range(n):
        y[i + 1] = y[i] + step_size * fun(x, y[i])
        x += step_size

    return y


if __name__ == "__main__":
    import timeit

    def f(x: float, y: float) -> float:
        return x + 0.6 * y

    X0 = 0.0
    Y0 = 0.2
    # If STEP_SIZE is smaller, the chart will be more accurate,
    # compare this chart with chart with STEP_SIZE = 0.2
    STEP_SIZE = 1.0
    X_END = 5

    print(
        "Execution time:",
        timeit.timeit(
            "euler_method(X0, Y0, STEP_SIZE, X_END, f)",
            number=1000,
            globals=globals(),
        ),
    )

    X = [round(X0 + STEP_SIZE * i, 2) for i in range(int((X_END - X0) / STEP_SIZE) + 1)]
    Y = euler_method(X0, Y0, STEP_SIZE, X_END, f)

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
