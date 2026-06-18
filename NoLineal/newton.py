import numpy as np
import sympy as sp

EPS = 1e-12


def newton(x, y, f, xn, t, epsilon):
    grad_f = sp.Matrix([sp.diff(f, x), sp.diff(f, y)])
    hess_f = sp.hessian(f, (x, y))
    grad_eval = sp.lambdify((x, y), grad_f, "numpy")
    hess_eval = sp.lambdify((x, y), hess_f, "numpy")

    print(f"{'k':>3} {'xn':>20} {'‖∇xn‖':>10} {'t':>10}  {'t_prima':>10}")
    print("-" * 60)

    k = 0
    while True:
        grad_xn_vec = np.array(grad_eval(*xn), dtype=float).flatten()
        hess_xn = np.array(hess_eval(*xn), dtype=float)
        xn1 = xn - t * np.linalg.solve(hess_xn, grad_xn_vec)
        grad_xn1_vec = np.array(grad_eval(*xn1), dtype=float).flatten()
        t_prima = t

        grad_xn = np.linalg.norm(grad_xn_vec)
        grad_xn1 = np.linalg.norm(grad_xn1_vec)

        if grad_xn1 < grad_xn:
            g0 = grad_xn1
            dif_grad = np.linalg.norm(grad_xn1_vec - grad_xn_vec)
            g_prima_0 = -grad_xn / dif_grad if dif_grad > EPS else 0
            g1 = grad_xn
            denom = 2 * (g1 - g0 - g_prima_0)
            beta = -g_prima_0 / denom if abs(denom) > EPS else 0.0
            t_prima = max(0.0, min(1.0, 1 - beta))
            xn1 = (1 - t_prima) * xn + t_prima * xn1

        print(f"{k:>3} {str(xn):>20} {grad_xn:>10.6f} {t:>10.6f}  {t_prima:>10.6f}")

        if np.linalg.norm(xn1 - xn) <= epsilon:
            print(f"\nx* = {xn1}")
            print(f"f(x*) = {float(f.subs({x: xn1[0], y: xn1[1]})):.6f}")
            return xn1

        xn = xn1
        k += 1


x, y = sp.symbols("x y")
f, xn, t, epsilon = (
    sp.sympify(input("ingrese f(x,y): ")),
    np.array(list(map(float, input("ingrese x0 (x,y): ").split(","))), dtype=float),
    float(input("ingrese t: ")),
    float(input("ingrese epsilon: ")),
)
newton(x, y, f, xn, t, epsilon)
