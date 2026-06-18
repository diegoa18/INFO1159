import sympy as sp
import numpy as np

x, y = sp.symbols('x y')

funcion_str = input("Ingrese la función f(x, y): ")
funcion = sp.sympify(funcion_str)

punto_actual = np.array(list(map(float, input("Ingrese el punto (separado por espacio): ").split())))
t = float(input("Ingrese t: "))
tolerancia = float(input("Ingrese la tolerancia: "))
n = int(input("Ingrese el número máximo de iteraciones: "))

gradiente = [sp.diff(funcion, x), sp.diff(funcion, y)]
grad_f = sp.lambdify((x, y), gradiente, "numpy")
    
iteracion = 0
gradiente_actual = np.array(grad_f(punto_actual[0], punto_actual[1]))
modulo_gradiente = sp.sqrt(gradiente_actual[0]**2 + gradiente_actual[1]**2)

while modulo_gradiente >= tolerancia and iteracion < n:
    
    delta_x = -gradiente_actual
    
    punto_actual = punto_actual + t * delta_x
    iteracion += 1
    
    gradiente_actual = np.array(grad_f(punto_actual[0], punto_actual[1]))
    modulo_gradiente = sp.sqrt(gradiente_actual[0]**2 + gradiente_actual[1]**2)

print(f"\nPunto optimo: {punto_actual}")
print(f"Iteraciones realizadas: {iteracion}")
