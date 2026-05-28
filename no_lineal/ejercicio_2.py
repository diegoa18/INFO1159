from sympy import symbols, sympify, lambdify
import numpy as np
import matplotlib.pyplot as plt
from math import sin,cos,pi

x = symbols('x')

f_input = input('Ingresa la funcion: ')
x_a = float(input('Ingresa xa: '))
x_b = float(input('Ingresa xb: '))
delta_lamda = float(input('Ingresa delta lambda: '))

expr = sympify(f_input)
f = lambdify(x, expr, 'numpy')

lamdas = np.arange(0, 1 + delta_lamda, delta_lamda)

def tipo_funcion():
    concava = False
    convexa = False

    for l in lamdas:
        punto_intermedio = l * x_a + (1 - l) * x_b
        f_del_promedio = f(punto_intermedio)
        promedio_de_f = l * f(x_a) + (1 - l) * f(x_b)

        if f_del_promedio <= promedio_de_f - 1e-9:
            convexa =True
        
        if f_del_promedio >= promedio_de_f + 1e-9:
            concava = True
    if concava and convexa:
        return "Ninguna de las dos"
    elif concava:
        return "Es concava"
    elif convexa:
        return "Es convexa"
    else:
        return "Ninguna de las dos"

resultado = tipo_funcion()
print(resultado)

t = np.linspace(x_a - 1, x_b + 1, 500)
plt.plot(t, f(t), "r-", label="f(x)")
plt.plot([x_a, x_b], [f(x_a), f(x_b)], "b--", label="recta")
plt.legend()
plt.show()