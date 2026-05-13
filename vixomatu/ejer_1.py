import numpy as np
import sympy as sp

x1, x2 = sp.symbols("x1 x2")
theta = sp.symbols("theta")


def fc(x, y, var):
    f = 1.20 * x + 1.16 * y - var * (2 * (x**2) + y**2 + (x + y) ** 2)
    return f


theta_value = float(input("Ingresa el valor de theta: "))

# Definir el rango para x1 y x2
Δx = np.arange(0, 5, 1e-2)

postulantes = []

# Búsqueda por fuerza bruta: evaluar en cada punto de la cuadrícula
for x in Δx:
    for y in Δx:
        if x + y <= 5:
            f_value = fc(x, y, theta_value)
            postulantes.append((f_value, x, y))

# se extraen los resultado y se encuentran los indices optimos
resultados_f = [tupla[0] for tupla in postulantes]
indice_optimo = np.argmax(resultados_f)

# se extraen los valores optimos
f_max, x1_opt, x2_opt = postulantes[indice_optimo]

print(f"Valor máximo de la función f(x*) = {f_max:.4f}")
print(f"En los puntos óptimos: x1* = {x1_opt:.4f}, x2* = {x2_opt:.4f}")
