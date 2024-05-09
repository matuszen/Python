import math
from collections.abc import Callable, Iterable


def euler_method(
    x0: float,
    y0: float,
    step_size: float,
    x_end: float,
    fun: Callable[[float, float], float],
) -> Iterable[float]:
    """
    Solves an initial value problem for an ordinary differential equation
    (ODE) using the Euler method.

    The Euler method is a first-order numerical procedure for solving
    ordinary differential equations (ODEs) with a given initial value.
    It is the simplest and most straightforward method for numerical integration.

    Parameters
    ----------
    x0 : float
        The initial value of the independent variable.
    y0 : float
        The initial value of the dependent variable.
    step_size : float
        The step size for the numerical integration. Must be positive.
    x_end : float
        The value of the independent variable at which to end the integration.
        Must be greater than `x0`.
    fun : (float, float) -> float
        The function that defines the ODE. It must take two arguments: the
        independent variable and the dependent variable.

    Returns
    -------
    Iterable[float]
        An iterable of the computed values of the dependent variable at each step.

    Raises
    ------
    ValueError
        If `step_size` is not positive or if `x_end` is not greater than `x0`.

    Notes
    -----
    The Euler method uses a fixed step size for integration and calculates the
    number of steps based on the range and step size. It updates the dependent
    variable by taking a step of size `step_size` in the direction of the slope
    defined by the function `fun`.

    Examples
    --------
    >>> def f(x, y):
    ...     return x + y
    >>> x0 = 0
    >>> y0 = 1
    >>> step_size = 0.1
    >>> x_end = 0.5
    >>> euler_method(x0, y0, step_size, x_end, f)
    [1.0, 1.1, 1.21, 1.331, 1.4641, 1.61051]
    """

    if step_size <= 0:
        raise ValueError("Step size must be positive.")

    if x_end <= x0:
        raise ValueError("x_end must be greater than x0.")

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

    import matplotlib.pyplot as plt

    def f(x: float, y: float) -> float:
        return x + 0.6 * y

    X0 = 0.0
    Y0 = 0.2
    # If STEP_SIZE is smaller, the chart will be more accurate,
    # compare this chart with chart with STEP_SIZE = 0.2
    STEP_SIZE = 0.5
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
