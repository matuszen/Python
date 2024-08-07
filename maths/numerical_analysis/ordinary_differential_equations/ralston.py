"""
Use the Ralston's method - Runge-Kutta 2nd order - to solve ODE.

https://math.libretexts.org/Workbench/Numerical_Methods_with_Applications_(Kaw)/8%3A_Ordinary_Differential_Equations/8.03%3A_Runge-Kutta_2nd-Order_Method_for_Solving_Ordinary_Differential_Equations#Ralston.E2.80.99s_Method
Author : Mateusz Nowak
"""

from collections.abc import Callable

import numpy as np


def ralston_method(
    f: Callable[[float, float], float],
    x0: float,
    y0: float,
    h: float,
    x_end: float,
) -> np.ndarray[tuple[int], float]:
    """
    Solves an ordinary differential equation using Ralston's method. Technically,
    it is practically the same method as Heun's method,
    but it minimizes the truncation error

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
    >>> ralston_method(f, x0, y0, h, x_end)
    array([1.        , 1.11      , 1.24205   , 1.39846525, 1.5818041 ,
           1.79489353])

    Notes
    -----
    Ralston's method is a second-order Runge-Kutta method that provides an improved
    estimate of the solution by averaging slopes at the beginning and an intermediate
    point within each step. It is known for its better accuracy compared to the
    standard Euler method while maintaining simplicity and minimizes truncation error.
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

    a1 = 1 / 3
    a2 = 2 / 3
    p1 = 3 / 4

    for i in range(n):
        k1 = f(x, y[i])
        k2 = f(x + p1 * h, y[i] + p1 * h * k1)
        y[i + 1] = y[i] + h * (a1 * k1 + a2 * k2)
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
            "ralston_method(f, X0, Y0, H, X_END)",
            number=1000,
            globals=globals(),
        ),
    )

    X = [round(X0 + H * i, 2) for i in range(int((X_END - X0) / H) + 1)]
    Y = ralston_method(f, X0, Y0, H, X_END)

    with plt.style.context("bmh"):
        fig, ax = plt.subplots()

        ax.plot(X, Y, "r", label="f(x, y) = x + 0.6 * y")

        ax.set_title("Ralston method")
        ax.set_xlabel("X Axis")
        ax.set_ylabel("Y Axis")

        fig.set_facecolor("#ddd")
        ax.set_facecolor("#ddd")

        plt.legend()
        plt.show()
