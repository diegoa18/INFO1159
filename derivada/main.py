from sympy import *

x = symbols('x')
expresion = input("ingrese la funcion: ")
puntox = float(input("ingrese el valor del punto x: "))
dx = float(input("ingrese el valor del punto dx: "))
eps = float(input("ingrese el valor de e: "))
expr = sympify(expresion)

def general(expr, x, num, eps):
    derivada = diff(expr, x)
    dx = sqrt(eps) * x

    if abs(derivada - num) < eps:
        return resultado

def numerica(expr, punto, deltax):
    fun = lambdify(x, expr, 'numpy')
    derivada = (fun(punto + deltax) - fun(punto)) / deltax
    return derivada

rn = numerica(expresion, x, dx)
rg = general(expresion, x, rn, eps)

print(f"numerica: {rn}, general: {rg}")
