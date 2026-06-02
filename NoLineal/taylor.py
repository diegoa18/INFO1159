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
    plt.grid()
    plt.show()


# x = sp.Symbol("x")
# func = sp.exp(x)
# plot_univariable(func, x, 0, 4)
