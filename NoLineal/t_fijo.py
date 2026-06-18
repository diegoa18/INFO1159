import numpy as np
import sympy as sp


def gradiente(variables, func):
    return sp.Matrix([sp.diff(func, v) for v in variables])


def descenso_fijo(variables, func, x0, t, eta=1e-6, max_iter=1000):
    xn = np.array(x0, dtype=float)
    grad_f = sp.lambdify(variables, gradiente(variables, func), "numpy")

    for _ in range(max_iter):
        grad_xn = np.array(grad_f(*xn), dtype=float).flatten()
        if np.linalg.norm(grad_xn) < eta:
            break
        delta_xn = -grad_xn
        xn = xn + t * delta_xn

    return xn


if __name__ == "__main__":
    x1, x2 = sp.symbols("x1 x2")
    f = sp.sympify(input("ingrese f(x1, x2): "))
    x0 = list(map(float, input("ingrese x0 (x1,x2): ").split(",")))
    t = float(input("ingrese t: "))

    x_opt = descenso_fijo([x1, x2], f, x0, t)
    print(f"x* = ({x_opt[0]:.6f}, {x_opt[1]:.6f})")
    print(f"f(x*) = {float(f.subs({x1: x_opt[0], x2: x_opt[1]})):.6f}")
