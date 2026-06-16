import numpy as np
import sympy as sp

x, y = sp.symbols("x y")
func = sp.sympify(input("ingrese una funcion objetivo: "))
t = float(input("ingrese un escalar: "))
x1, x2 = map(float, input("ingrese el punto inicial (x y): ").split())
eps = float(input("ingrese un valor de error (0, 1]: "))

grad_num = sp.lambdify(
    (x, y), sp.sympify([sp.diff(func, x), sp.diff(func, y)]), "numpy"
)
func_num = sp.lambdify((x, y), func, "numpy")


def gradescendente_barzilai(x1, x2, t, eps):
    xn = np.array([x1, x2], dtype=float)
    iteracion = 0
    gradiente = np.array(grad_num(xn[0], xn[1]), dtype=float)

    xn_anterior = xn.copy()
    grad_anterior = gradiente.copy()

    while np.linalg.norm(gradiente, ord=2) >= eps:  # norma euclidiana
        delta = -gradiente
        if iteracion > 0:
            s = xn - xn_anterior
            y_bb = gradiente - grad_anterior
            if np.linalg.norm(y_bb) > 0:
                t = abs((np.dot(s, y_bb)) / np.linalg.norm(y_bb) ** 2)

        xn_anterior = xn.copy()
        grad_anterior = gradiente.copy()

        xn = xn + t * delta

        iteracion += 1

        print(
            f"Iteracion {iteracion}: x={xn[0]:.3f}, y={xn[1]:.3f}, t={t:.3f}, f={func_num(xn[0], xn[1]):.3f} "
        )

        gradiente = np.array(grad_num(xn[0], xn[1]), dtype=float)

    return xn


resultado = gradescendente_barzilai(x1, x2, t, eps)

print("\nPunto óptimo aproximado:")
print(f"x =, {resultado[0]:.6f}")
print(f"y =, {resultado[1]:.6f}")
print(f"f(x,y) = {func_num(resultado[0], resultado[1]):.6f}")
