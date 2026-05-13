from sympy import symbols, sympify, lambdify
import numpy as np

x1, x2, theta_sym = symbols('x1 x2 theta')

print("Ingrese la funcion en terminos de x1, x2, theta ")
print("Ejemplo: 1.2*x1 + 1.16*x2 - theta*(2*x1**2 + x2**2 + (x1 + x2)**2)")
expresion_str = input("Funcion: ")

try:
    # Convertir string a expresión simbólica
    expresion = sympify(expresion_str)
    # Crear función evaluable
    funcion = lambdify((x1, x2, theta_sym), expresion, 'numpy')
except Exception as e:
    print(f"Error al procesar la función: {e}")
    exit()

try:
    theta = float(input("Ingrese el valor de theta: "))
except ValueError:
    print("Error: theta debe ser un número.")
    exit()

lista = []
delta_x = 0.1
dx = np.arange(0,5 + delta_x, delta_x)

for x_1 in dx:
    for x_2 in dx:
        if x_1 + x_2 <= 5:
            valor = funcion(x_1, x_2, theta)
            lista.append(((x_1, x_2), valor))

mejor_tupla = max(lista, key=lambda item: item[1]) #gpt

x_estrella = mejor_tupla[0]
f_x_estrella = mejor_tupla[1]

print(f" Resultados con theta = {theta}")
print(f"x* (x1, x2): {x_estrella}")
print(f"f(x*): {f_x_estrella}")