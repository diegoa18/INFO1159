import matplotlib.pyplot as plt
import numpy as np
import sympy as sp


def multivariable(variables, func, x_vals, orden=2):
    a = dict(zip(variables, x_vals))
    delta = sp.Matrix([v - a_v for v, a_v in a.items()])
    aprox = func.subs(a)
    grad = sp.Matrix([sp.diff(func, v) for v in variables])
    aprox += grad.subs(a).dot(delta)
    if orden >= 2:
        H = sp.hessian(func, variables)
        aprox += (delta.T * H.subs(a) * delta)[0] / 2
    return sp.expand(aprox)


# SOLO GRAFICA DOS VARIABLES
def plot_multivariable(variables, func, x_vals, orden=2, rango=5):
    poly = multivariable(variables, func, x_vals, orden)
    f_n = sp.lambdify(variables, func, "numpy")
    p_n = sp.lambdify(variables, poly, "numpy")

    if len(variables) == 2:
        a, b = x_vals
        xs = np.linspace(a - rango, a + rango, 100)
        ys = np.linspace(b - rango, b + rango, 100)
        X, Y = np.meshgrid(xs, ys)
        Z_f, Z_p = f_n(X, Y), p_n(X, Y)

        fig = plt.figure(figsize=(12, 5))
        ax1 = fig.add_subplot(121, projection="3d")
        ax1.plot_surface(X, Y, Z_f, cmap="viridis", alpha=0.8)
        ax1.set_title("$f(x, y)$")
        ax2 = fig.add_subplot(122, projection="3d")
        ax2.plot_surface(X, Y, Z_p, cmap="plasma", alpha=0.8)
        ax2.set_title(f"taylor orden {orden}")
        plt.show()
    else:
        print("plot_multivariable: solo soportado para 2 variables")
