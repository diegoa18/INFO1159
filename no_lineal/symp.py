import sympy as sp

def derivar(funcion):
    x = sp.Symbol('x')

    funcion = x**2 + 3
    derivada = sp.diff(funcion, x)
    return derivada

print(derivar())