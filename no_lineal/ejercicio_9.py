import sympy as sp
import numpy as np

x, y = sp.symbols('x y')

funcion_str = input("Ingrese la función f(x, y): ")
funcion = sp.sympify(funcion_str)

punto_actual = np.array(list(map(float, input("Ingrese el punto (separado por espacio): ").split())))

t = float(input("Ingrese t: "))
tolerancia = float(input("Ingrese la tolerancia: "))
n = int(input("Ingrese el número máximo de iteraciones: "))

gradiente = [sp.diff(funcion, x),sp.diff(funcion, y)]
grad_f = sp.lambdify((x, y), gradiente, "numpy")

iteracion = 0
while iteracion < n:

    gradiente_actual = np.array(grad_f(punto_actual[0], punto_actual[1]), dtype= float)

    modulo_gradiente = np.linalg.norm(gradiente_actual)

    if iteracion > 0:

        s_k = punto_actual - punto_anterior
        y_k = gradiente_actual - gradiente_anterior

        denominador = np.dot(s_k, y_k)

        if abs(denominador) < 1e-12:
            print("Denominador demasiado pequeño")
            break

        t = np.dot(s_k, s_k) / denominador

    punto_anterior = punto_actual.copy()
    gradiente_anterior = gradiente_actual.copy()

    punto_actual = punto_actual - t * gradiente_actual

    iteracion += 1

print(f"Punto óptimo: {punto_actual}")
print(f"Iteraciones: {iteracion}")