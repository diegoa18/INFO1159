from sympy import *
import math

x = symbols('x')

expresion = input("Ingrese la función: ")
puntox = float(input("Ingrese el valor del punto x: "))

eps = 2.22e-16
dx = math.sqrt(eps) * puntox

expr = sympify(expresion)
# Derivada exacta
def general(expr, variable, punto):
    derivada = diff(expr, variable)
    resultado = derivada.subs(variable, punto)
    return resultado

# Derivada numérica
def numerica(expr, punto, deltax):
    fun = lambdify(x, expr, 'numpy')
    derivada = (fun(punto + deltax) - fun(punto)) / deltax
    return derivada


# Resultados
rn = numerica(expr, puntox, dx)
rg = general(expr, x, puntox)
error = rn-rg

print(f"Derivada numérica: {rn}")
print(f"Derivada exacta: {rg}")
if error < eps:
    print(f"el error: {error} no es menor que {eps}")
else:
    print(f"el error: {error} es menor que eps")
