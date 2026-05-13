import numpy as np
import sympy as sp

theta = float(input("Ingrese el valor de θ: "))
dx = float(input("Ingrese el valor para Δx: "))

rango = np.arange(0, 5000 + dx, dx)

x1, x2 = sp.symbols("x1 x2")
expresion_obj = 1.20 * x1 + 1.16 * x2 - theta * (2 * x1**2 + x2**2 + (x1 + x2) ** 2)
obj = sp.lambdify((x1, x2), expresion_obj, "numpy")

evaluaciones = []

for x1 in rango:
    for x2 in rango:
        if x1 + x2 <= 5000:
            valor = obj(x1, x2)
            evaluaciones.append(([x1, x2], valor))

valores_obj = [t[1] for t in evaluaciones]
indice_opt = np.argmax(valores_obj)
optimo = evaluaciones[indice_opt]

print(f"x^* = ({optimo[0][0]}, {optimo[0][1]})")
print(f"f(x^*) = max f(x) = {optimo[1]}")
