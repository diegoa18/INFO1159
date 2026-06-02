import numpy as np
import sympy as sp

vars_str = input("ingrese las variables separadas por espacio: ")
variables = sp.symbols(vars_str.split())

const_str = input("ingrese constantes separadas por espacio (enter si no hay): ")
constantes = sp.symbols(const_str.split()) if const_str else []

funcion_str = input("ingrese la funcion: ")
funcion = sp.sympify(funcion_str)

if constantes:
    for c in constantes:
        val_c = float(input(f"ingrese el valor de la constante {c}: "))
        funcion = funcion.subs(c, val_c)

punto_str = input("ingrese el vector x a evaluar (separado por espacio): ")
vector_x = np.array([float(val) for val in punto_str.split()])

delta_str = input("ingrese el vector delta x (separado por espacio): ")
vector_dx = np.array([float(val) for val in delta_str.split()])

f_num = sp.lambdify(variables, funcion, "numpy")


def calcular_gradiente_numerico(f, punto, deltas):
    # calculamos el gradiente perturbando una variable a la vez
    gradiente = []
    val_original = f(*punto)

    for i in range(len(punto)):
        punto_perturbado = punto.copy()
        punto_perturbado[i] = punto_perturbado[i] + deltas[i]
        val_perturbado = f(*punto_perturbado)

        derivada_parcial = (val_perturbado - val_original) / deltas[i]
        gradiente.append(derivada_parcial)

    return np.array(gradiente)


grad_num = calcular_gradiente_numerico(f_num, vector_x, vector_dx)

print(f"gradiente numerico con delta x : {grad_num}")
