import numpy as np
import sympy as sp

eps = np.finfo(float).eps
x = sp.symbols("x")

funcion_str = input("ingrese la funcion: ")
punto = float(input("ingrese el punto a evaluar: "))
error = float(input("ingrese el valor de error: "))

if error < eps:
    print(f"el error debe ser mayor o igual al epsilon de la maquina ({eps})")
    exit()

funcion = sp.sympify(funcion_str)
f_num = sp.lambdify(x, funcion)


def calcular_derivada_explicita(funcion, var, punto_eva):
    df = sp.diff(funcion, var)
    dfx = sp.lambdify(var, df)
    return dfx(punto_eva)


def calcular_derivada_definicion(f_num, punto_eva, error, derivada_explicita):
    # iteramos para encontrar un delta x que cumpla la condicion
    delta_x = 1.0
    for _ in range(100):
        derivada_def = (f_num(punto_eva + delta_x) - f_num(punto_eva)) / delta_x
        if abs(derivada_explicita - derivada_def) < error:
            return derivada_def, delta_x
        delta_x = delta_x / 2.0

    return derivada_def, delta_x


def comparar_derivadas():
    # calculamos primero la explicita para poder usarla en la definicion
    derivada_explicita = calcular_derivada_explicita(funcion, x, punto)
    derivada_definicion, delta_x = calcular_derivada_definicion(
        f_num, punto, error, derivada_explicita
    )

    print(f"derivada explicita : {derivada_explicita}")
    print(f"derivada por definicion : {derivada_definicion}")
    print(f"delta x encontrado : {delta_x}")

    if abs(derivada_explicita - derivada_definicion) < error:
        return True
    return False


print(comparar_derivadas())
