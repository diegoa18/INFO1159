import matplotlib.pyplot as plt
import numpy as np
import sympy as sp


def univariable(func, x, t, n=2):
    polinomio = 0
    for k in range(n + 1):
        coeficiente = sp.diff(func, x, k).subs(x, t)
        termino = coeficiente * (x - t) ** k / sp.factorial(k)
        polinomio += termino
    return sp.expand(polinomio)


def plot_univariable(func, x, t, xa, xb, dx, n=2):
    polinomio = univariable(func, x, t, n)
    func_n, poli_n = sp.lambdify(x, func, "numpy"), sp.lambdify(x, polinomio, "numpy")
    xs = np.linspace(xa, xb, int(round((xb - xa) / dx)) + 1)
    plt.plot(xs, func_n(xs), "r", label="f(x)")
    plt.plot(xs, poli_n(xs), "b", label="Pn(x)")
    plt.legend()
    plt.show()


x = sp.Symbol("x")
f, t_val, xa, xb, dx = (
    sp.sympify(input("ingrese f(x): ")),
    float(input("Ingrese el punto de expansion t: ")),
    float(input("xa: ")),
    float(input("xb: ")),
    float(input("ingrese el paso dx: ")),
)
plot_univariable(f, x, t_val, xa, xb, dx)
