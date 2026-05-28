import sympy as sp

x = sp.symbols('x')

funcion = sp.sympify(input("Ingresa la función f(x): "))
valor_x = float(input("Ingresa el valor de x donde evaluar: "))
delta_x = float(input("Ingresa el valor de delta x (ej: 0.001): "))


f = sp.lambdify(x, funcion, 'numpy')

def encontrar_derivada_numerica(f, valor_x, delta_x):
    return (f(valor_x + delta_x) - f(valor_x)) / delta_x

derivada = encontrar_derivada_numerica(f, valor_x, delta_x)

print(derivada)