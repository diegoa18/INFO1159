import sympy as sp
import numpy as np

x = sp.symbols('x')

funcion_str = input("Ingrese la función f(x): ")
funcion = sp.sympify(funcion_str)

punto_actual = float(input("Ingrese el punto inicial x0: "))
t_inicial = float(input("Ingrese t > 0: "))
epsilon = float(input("Ingrese epsilon: "))
max_iter = int(input("Ingrese el número máximo de iteraciones: "))

primera_derivada = sp.diff(funcion, x)
segunda_derivada = sp.diff(funcion, x, x)

f_deriv1 = sp.lambdify(x, primera_derivada, "numpy")
f_deriv2 = sp.lambdify(x, segunda_derivada, "numpy")

iteracion = 0
while iteracion < max_iter:
    d1_actual = float(f_deriv1(punto_actual))
    d2_actual = float(f_deriv2(punto_actual))
    
    g_actual_norm = abs(d1_actual)
    
    if abs(d2_actual) < 1e-12:
        print("La segunda derivada es cercana a cero. El metodo de Newton no puede continuar")
        break
        
    t = t_inicial
    
    # Paso 2.1
    paso_newton = d1_actual / d2_actual
    punto_siguiente = punto_actual - t * paso_newton
    
    d1_siguiente = float(f_deriv1(punto_siguiente))
    g_siguiente_norm = abs(d1_siguiente)
    
    # Paso 2.2
    if g_siguiente_norm > g_actual_norm:
        
        # Paso 2.2.1
        g_0 = g_actual_norm
        
        # Paso 2.2.2
        g_prima_0 = -g_actual_norm * abs(d1_siguiente - d1_actual)
        
        # Paso 2.2.3
        g_1 = g_siguiente_norm
        
        # Paso 2.2.4
        denominador = 2.0 * (g_1 - g_0 - g_prima_0)
        if abs(denominador) > 1e-12:
            t0 = -g_prima_0 / denominador
            t0 = max(t0, 0.1)
        else:
            t0 = 0.1
            
        # Paso 2.2.5
        punto_siguiente = (1.0 - t0) * punto_actual + t0 * punto_siguiente
    
    distancia = abs(punto_siguiente - punto_actual)
    
    punto_actual = punto_siguiente
    iteracion += 1
    
    if distancia <= epsilon:
        print(f"\nConvergió por criterio de tolerancia de paso.")
        break

print(f"\nPunto óptimo estimado: {punto_actual}")
print(f"Iteraciones realizadas: {iteracion}")
print(f"Derivada primera final: {f_deriv1(punto_actual)}")