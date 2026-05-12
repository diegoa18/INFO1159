from random import uniform

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp

lda = uniform(0, 1)
func_input = input("ingrese la funcion no lineal para x: ")
xa = float(input("ingrese el valor para xa: "))
xb = float(input("ingrese el valor para xb: "))

x = sp.symbols("x")
func_expression = sp.sympify(func_input)
func = sp.lambdify((x), func_expression, "math")


if func(lda * xa + (1 - lda) * xb) >= lda * func(xa) + (1 - lda) * func(xb):
    print("en el intervalo [xa,xb] la funcion es concava")
if func(lda * xa + (1 - lda) * xb) <= lda * func(xa) + (1 - lda) * func(xb):
    print("en el intervalo [xa,xb] la funcion es convexa")

x_values = [xa, xb]
y_values = [func(xa), func(xb)]

rangex = np.arange(xa - 1, xb + 1, 0.1)
rangey = func(rangex)

plt.plot(x_values, y_values, "bo", linestyle="-", color="blue")
plt.plot(rangex, rangey, color="red")

plt.show()
