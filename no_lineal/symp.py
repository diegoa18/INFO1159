import sympy as sp
from sympy import Symbol

def derivar():
    x = sp.Symbol('x')
    
    funcion = x**2 + 3
    derivada = sp.diff(funcion)
    return derivada

print(derivar)