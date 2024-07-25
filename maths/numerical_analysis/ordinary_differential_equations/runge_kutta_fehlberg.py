from collections.abc import Callable

import numpy as np


def runge_kutta_fehlberg_method(
    f: Callable[[float, float], float],
    x0: float,
    y0: float,
    h: float,
    x_end: float,
) -> np.ndarray[tuple[int], float]:
    """
    Solves an ordinary differential equation using the Runge-Kutta-Fehlberg method
    (RKF 45).

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
        If `x0` is greater than or equal to `x_end`.

    Examples
    --------
    >>> def f(x, y):
    ...     return x + y
    >>> x0 = 0
    >>> y0 = 1
    >>> h = 0.1
    >>> x_end = 0.5
    >>> runge_kutta_fehlberg_method(f, x0, y0, h, x_end)
    array([1.        , 1.11034183, 1.24280551, 1.39971761, 1.58364939,
           1.79744253])

    Notes
    -----
    The Runge-Kutta-Fehlberg method is an adaptive method for solving ordinary
    differential equations. It uses an embedded Runge-Kutta method to estimate
    the error and adjust the step size accordingly. This implementation uses a
    fixed step size but employs the coefficients of the Runge-Kutta-Fehlberg
    method to compute the solution.
    """

    if h <= 0:
        raise ValueError("Step size must be positive")

    if x0 >= x_end:
        raise ValueError(
            "The final value of x must be greater than the first value of x"
        )

    n = int((x_end - x0) / h)
    y = np.zeros(n + 1)

    x = x0
    y[0] = y0

    # Coefficients on Butcher tableau
    # see https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta%E2%80%93Fehlberg_method
    a = [0, 1 / 4, 3 / 8, 12 / 13, 1, 1 / 2]
    b = [
        [],
        [1 / 4],
        [3 / 32, 9 / 32],
        [1932 / 2197, -7200 / 2197, 7296 / 2197],
        [439 / 216, -8, 3680 / 513, -845 / 4104],
        [-8 / 27, 2, -3544 / 2565, 1859 / 4104, -11 / 40],
    ]
    c = [16 / 135, 0, 6656 / 12825, 28561 / 56430, -9 / 50, 2 / 55]

    for i in range(n):
        k1 = h * f(x, y[i])
        k2 = h * f(x + a[1] * h, y[i] + b[1][0] * k1)
        k3 = h * f(x + a[2] * h, y[i] + b[2][0] * k1 + b[2][1] * k2)
        k4 = h * f(x + a[3] * h, y[i] + b[3][0] * k1 + b[3][1] * k2 + b[3][2] * k3)
        k5 = h * f(
            x + a[4] * h,
            y[i] + b[4][0] * k1 + b[4][1] * k2 + b[4][2] * k3 + b[4][3] * k4,
        )
        k6 = h * f(
            x + a[5] * h,
            y[i]
            + b[5][0] * k1
            + b[5][1] * k2
            + b[5][2] * k3
            + b[5][3] * k4
            + b[5][4] * k5,
        )
        y[i + 1] = (
            y[i] + c[0] * k1 + c[1] * k2 + c[2] * k3 + c[3] * k4 + c[4] * k5 + c[5] * k6
        )
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
            "runge_kutta_fehlberg_method(f, X0, Y0, H, X_END)",
            number=1000,
            globals=globals(),
        ),
    )

    X = [round(X0 + H * i, 2) for i in range(int((X_END - X0) / H) + 1)]
    Y = runge_kutta_fehlberg_method(f, X0, Y0, H, X_END)

    with plt.style.context("bmh"):
        fig, ax = plt.subplots()

        ax.plot(X, Y, "r", label="f(x, y) = x + 0.6 * y")

        ax.set_title("Runge-Kutta-Fehlberg method")
        ax.set_xlabel("X Axis")
        ax.set_ylabel("Y Axis")

        fig.set_facecolor("#ddd")
        ax.set_facecolor("#ddd")

        plt.legend()
        plt.show()
