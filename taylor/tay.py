import matplotlib.pyplot as plt
import numpy as np
import sympy as sp

x = sp.Symbol("x")

f = sp.sympify(input("ingrese una funcion:"))
xa = float(input("ingrese xa: "))
xb = float(input("ingrese xb: "))
dx = float(input("ingrese dx: "))
t = float(input("ingrese t: "))


def univariable(f, x, t, n=2):
    p = 0
    for k in range(n + 1):
        d1 = sp.diff(f, x, k).subs(x, t)
        p += d1 * (x - t) ** k / sp.factorial(k)
    return sp.expand(p)


def graficar(f, x, t, xa, xb, dx, n=2):
    p = univariable(f, x, t, n)
    f_num, tay_num = sp.lambdify(x, f, "numpy"), sp.lambdify(x, p, "numpy")
    xs = np.linspace(xa, xb, int(round((xb - xa) / dx)) + 1)
    plt.plot(xs, f_num(xs), "r", label="f(x)")
    plt.plot(xs, tay_num(xs), "b", label="Pn(x)")
    plt.legend()
    plt.show()


graficar(f, x, t, xa, xb, dx)
