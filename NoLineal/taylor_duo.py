import numpy as np
import sympy as sp


def hessiana_num(func, vars, x0, dx):
    f = sp.lambdify(vars, func, "numpy")
    a, b = x0[0], x0[1]
    h1, h2 = dx[0], dx[1]

    f_xx = (f(a + h1, b) - 2 * f(a, b) + f(a - h1, b)) / h1**2
    f_yy = (f(a, b + h2) - 2 * f(a, b) + f(a, b - h2)) / h2**2
    f_xy = (
        f(a + h1, b + h2) - f(a + h1, b - h2) - f(a - h1, b + h2) + f(a - h1, b - h2)
    ) / (4 * h1 * h2)

    return f_xx, f_yy, f_xy


def clasificar_punto(f_xx, f_yy, f_xy):
    H = f_xx * f_yy - f_xy**2
    if abs(H) < 1e-10:
        return "Sin informacion suficiente (H = 0)"
    if H < 0:
        return "Punto de silla"
    if f_xx < 0:
        return "Maximo local"
    return "Minimo local"


x1, x2 = sp.symbols("x1 x2")
f, x1_val, x2_val, dx1, dx2 = (
    sp.sympify(input("ingrese f(x1, x2): ")),
    float(input("x1: ")),
    float(input("x2: ")),
    float(input("dx1: ")),
    float(input("dx2: ")),
)
f_xx, f_yy, f_xy = hessiana_num(f, [x1, x2], [x1_val, x2_val], [dx1, dx2])
resultado = clasificar_punto(f_xx, f_yy, f_xy)
print(f"f_xx = {f_xx:.6f}, f_yy = {f_yy:.6f}, f_xy = {f_xy:.6f}")
print(f"H(x) = {f_xx * f_yy - f_xy**2:.6f}")
print(f"({x1_val}, {x2_val}) -> {resultado}")
