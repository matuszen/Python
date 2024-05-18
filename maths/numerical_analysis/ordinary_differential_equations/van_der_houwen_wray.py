from collections.abc import Callable, Iterable

import numpy as np


def van_der_houwen_wray_method(
    f: Callable[[float, float], float],
    x0: float,
    y0: float,
    h: float,
    x_end: float,
) -> Iterable[float]:
    """
    Solves an ordinary differential equation using the Van der Houwen-Wray method.

    Parameters
    ----------
    f : Callable[[float, float], float]
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
    Iterable[float]
        Array of y values at each step from x0 to x_end.

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
    >>> van_der_houwen_wray_method(f, x0, y0, h, x_end)
    array([1.        , 1.11033333, 1.24278672, 1.39968646, 1.58360349,
           1.79737912])

    Notes
    -----
    The Van der Houwen-Wray method is a third-order Runge-Kutta method. It uses a
    specific set of coefficients defined in the Butcher tableau to achieve higher
    accuracy compared to first- and second-order methods. This method is particularly
    useful for stiff equations where higher-order methods are beneficial.
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

    # Coefficients from Butcher tableau
    # see https://en.wikipedia.org/wiki/List_of_Runge%E2%80%93Kutta_methods#Van_der_Houwen's/Wray_third-order_method
    a = [0, 1 / 3, 2 / 3]
    b = [
        [0, 0, 0],
        [1 / 3, 0, 0],
        [0, 2 / 3, 0],
    ]
    c = [1 / 4, 0, 3 / 4]

    for i in range(n):
        # Simplify this line for better computing efficiency
        # k1 = f(x + h * a[0], y[i] + h * b[0][0])
        k1 = f(x, y[i])
        k2 = f(x + h * a[1], y[i] + h * b[1][0] * k1)
        k3 = f(x + h * a[2], y[i] + h * b[2][1] * k2)

        y[i + 1] = y[i] + h * (c[0] * k1 + c[2] * k3)
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
            "van_der_houwen_wray_method(f, X0, Y0, H, X_END)",
            number=1000,
            globals=globals(),
        ),
    )

    X = [round(X0 + H * i, 2) for i in range(int((X_END - X0) / H) + 1)]
    Y = van_der_houwen_wray_method(f, X0, Y0, H, X_END)

    with plt.style.context("bmh"):
        fig, ax = plt.subplots()

        ax.plot(X, Y, "r", label="f(x, y) = x + 0.6 * y")

        ax.set_title("Van der Houwen-Wray method")
        ax.set_xlabel("X Axis")
        ax.set_ylabel("Y Axis")

        fig.set_facecolor("#ddd")
        ax.set_facecolor("#ddd")

        plt.legend()
        plt.show()
