import math
from collections.abc import Callable, Iterable


def modified_euler_method(
    x0: float,
    y0: float,
    h: float,
    x_end: float,
    fun: Callable[[float, float], float],
) -> Iterable[float]:
    """
    Solves an initial value problem for an ordinary differential equation (ODE)
    using the Modified Euler method.

    The Modified Euler method, also known as Heun's method, is a numerical technique
    used to solve first-order ODEs. It improves the accuracy of
    the standard Euler method by incorporating an average of the slopes at
    the beginning and the end of the interval.

    Parameters
    ----------
    x0 : float
        The initial value of the independent variable.
    y0 : float
        The initial value of the dependent variable.
    h : float
        The step size for the numerical integration. Must be positive.
    x_end : float
        The value of the independent variable at which to end the integration.
        Must be greater than `x0`.
    fun : (float, float) -> float
        The function that defines the ODE. It must take two arguments:
        the independent variable and the dependent variable.

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
    The function uses a fixed step size for integration and calculates
    the number of steps based on the range and step size.
    The Modified Euler method involves two stages:
    1. An initial estimate of the slope (k1) at the beginning of the interval.
    2. A corrected slope (k2) that uses the midpoint value.

    Examples
    --------
    >>> def f(x, y):
    ...     return x + y
    >>> x0 = 0
    >>> y0 = 1
    >>> step_size = 0.1
    >>> x_end = 0.5
    >>> modified_euler_method(x0, y0, step_size, x_end, f)
    [1.0, 1.105, 1.221025, 1.349075125, 1.49026888125, 1.645832125625]
    """

    if h <= 0:
        raise ValueError("Step size must be positive.")

    if x_end <= x0:
        raise ValueError("x_end must be greater than x0.")

    n = math.ceil((x_end - x0) / h)

    y = [0 for _ in range(n + 1)]
    y[0] = y0
    x = x0

    for i in range(n):
        k1 = (x + x + h) * 0.5
        k2 = y[i] + h * fun(x, y[i]) * 0.5
        y[i + 1] = y[i] + h * fun(k1, k2)

        x += h

    return y


if __name__ == "__main__":
    import timeit

    import matplotlib.pyplot as plt

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
            "modified_euler_method(X0, Y0, H, X_END, f)",
            number=1000,
            globals=globals(),
        ),
    )

    X = [round(X0 + H * i, 2) for i in range(int((X_END - X0) / H) + 1)]
    Y = modified_euler_method(X0, Y0, H, X_END, f)

    with plt.style.context("bmh"):
        fig, ax = plt.subplots()

        ax.plot(X, Y, "r", label="f(x, y) = x + 0.6 * y")

        ax.set_title("Modified euler method")
        ax.set_xlabel("X Axis")
        ax.set_ylabel("Y Axis")

        fig.set_facecolor("#ddd")
        ax.set_facecolor("#ddd")

        plt.legend()
        plt.show()
