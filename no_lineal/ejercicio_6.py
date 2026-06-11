import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

def graficar_taylor(funcion, derivada1, derivada2,xa, xb, delta_x, t, variable):
    f = sp.lambdify(variable, funcion, "numpy")
    d1 = sp.lambdify(variable, derivada1, "numpy")
    d2 = sp.lambdify(variable, derivada2, "numpy")

    x = np.linspace(xa, xb, 1000)

    y_real = f(x + delta_x)

    y_taylor = ( f(x) + d1(x) * delta_x + 0.5 * d2(x + t * delta_x) * (delta_x ** 2))

    plt.plot(x, y_real, color="red", linewidth=2, label=r"$f(x+\Delta x)$")
    plt.plot(x, y_taylor, color="blue", linestyle="--", linewidth=2, label="Taylor de segundo orden")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    x = sp.Symbol('x')

    funcion_str = input("Ingrese la función f(x): ")
    funcion = sp.sympify(funcion_str)

    derivada1 = sp.diff(funcion, x)
    derivada2 = sp.diff(derivada1, x)

    xa = float(input("\nIngrese xa: "))
    xb = float(input("Ingrese xb: "))

    delta_x = float(input("Ingrese delta_x: "))
    t = float(input("Ingrese t: "))

    graficar_taylor(funcion, derivada1, derivada2, xa, xb, delta_x, t, x)