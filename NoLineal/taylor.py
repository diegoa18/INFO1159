"""UNIVARIABLE
sumatoria desde k = 0 hasta n para (f**k(a)/k!)*(x-a)**k
donde k seria el orden de la derivada de la funcion.
para f(a) se reemplaza x por el valor de a, pero para (x - a), x es simbolica
esto para evaluar el polinomio.
a = punto de expansion -> donde se construye la aproximacion
donde obligamos al polinomio Pn(x) parecerse a la funcion"""

"""MULTIVARIABLE
t parametriza el segmento entre xn y xn + deltaxn
f(xn + deltaxn) = f(xn) + gradf(xn + t * deltaxn)trans * deltaxn
siendo el cambio de un f(x) cuando nos movemos de x hasta x + deltax
respecto al gradiente, cuanto cambia el grad cuando nos movemos de xn a deltaxn:
    gradf(xn+deltaxn) = gradf(xn) + integral desde 0 a 1 de grad2f(xn + t*deltaxn)deltax dt
    la integral siempre de 0 a 1 y el grad2 ya que es la matriz hessiana (derivadas segundo orden)

esto equivalente a:
    f(xn + deltaxn) = f(xn) + gradf(xn)trans*deltaxn + ((1/2)*deltaxn)trans*grad2f(xn + t*deltaxn)deltaxn"""

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp


def univariable(func, x, a, n):
    polinomio = 0
    for k in range(n + 1):
        derivada = sp.diff(func, x, k)
        coeficiente = derivada.subs(x, a)
        termino = coeficiente * (x - a) ** k / sp.factorial(k)
        polinomio += termino
    return sp.expand(polinomio)


def plot_univariable(func, x, a, n):
    polinomio = univariable(func, x, a, n)
    func_n, poli_n = sp.lambdify(x, func, "numpy"), sp.lambdify(x, polinomio, "numpy")
    xs = np.linspace(a - 5, a + 5, 1000)
    plt.plot(xs, func_n(xs), label="f(x)")
    plt.plot(xs, poli_n(xs), label="Pn(x)")
    plt.legend()
    plt.show()


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
