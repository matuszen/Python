from collections.abc import Callable

import numpy as np


def heun_method(
    f: Callable[[float, float], float],
    x0: float,
    y0: float,
    h: float,
    x_end: float,
) -> np.ndarray[tuple[int], float]:
    """
    Solves an ordinary differential equation using the Heun's method (modified Euler).

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
    >>> heun_method(f, x0, y0, h, x_end)
    array([1.        , 1.11      , 1.24205   , 1.39846525, 1.5818041 ,
           1.79489353])

    Notes
    -----
    The Heun's method, also known as modified Euler method, is a numerical procedure
    to solve ordinary differential equations of the form dy/dx = f(x, y). This method
    improves the accuracy of the standard Euler method by using the average of the
    slopes at the beginning and end of the step to compute the next value of y.
    """

    if h <= 0:
        raise ValueError("Step size must be positive")

    if x_end <= x0:
        raise ValueError(
            "The final value of x must be greater than the first value of x"
        )

    n = int((x_end - x0) / h)
    y = np.zeros((n + 1,))

    x = x0
    y[0] = y0

    for i in range(n):
        k1 = (x + x + h) * 0.5
        k2 = y[i] + h * f(x, y[i]) * 0.5
        y[i + 1] = y[i] + h * f(k1, k2)
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
    H = 1
    X_END = 10

    print(
        "Execution time:",
        timeit.timeit(
            "heun_method(f, X0, Y0, H, X_END)",
            number=1000,
            globals=globals(),
        ),
    )

    X = [round(X0 + H * i, 2) for i in range(int((X_END - X0) / H) + 1)]
    Y = heun_method(f, X0, Y0, H, X_END)

    with plt.style.context("bmh"):
        fig, ax = plt.subplots()

        ax.plot(X, Y, color="red", label="f(x, y) = x + 0.6 * y")
        ax.plot(X, Y, color="red", marker="o", linewidth=1, label="Calculated points")

        ax.set_title("Heun's method")
        ax.set_xlabel("X Axis")
        ax.set_ylabel("Y Axis")

        fig.set_facecolor("#ddd")
        ax.set_facecolor("#ddd")

        plt.legend()
        plt.show()
