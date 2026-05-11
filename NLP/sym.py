from sympy import symbols, sympify, lambdify, Piecewise, Eq
import numpy as np
x, y = symbols('x y')
expr = input("Ingrese la función a optimizar: ")
expr = sympify(expr)

print(expr - 2*x)

func = lambdify(x, Piecewise((0, Eq(x, 0)), (x + 1, True)))
print(func(np.arange(5)))