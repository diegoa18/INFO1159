import sympy as sp

x = sp.symbols('x')

funcion = sp.sympify(input("Ingresa la función f(x): "))
valor_x = float(input("Ingresa el valor de x donde evaluar: "))
epsilon = float(input("Ingresa el valor del error absoluto (ej: 0.001): "))

if epsilon <= 0:
    print("Epsilon debe ser mayor que 0")
    exit()

f = sp.lambdify(x, funcion, 'numpy')

derivada_simbolica = sp.diff(funcion, x)
f_derivada_exacta = sp.lambdify(x, derivada_simbolica, 'numpy')
valor_derivada_exacta = f_derivada_exacta(valor_x)
""" 
def encontrar_derivada_numerica(f, valor_x, delta_x):
    return (f(valor_x + delta_x) - f(valor_x)) / delta_x """

# 5. Algoritmo de Búsqueda Binaria para encontrar Delta X
def buscar_delta_x(f, valor_x, valor_exacto, epsilon):
    bajo = 1e-15 
    alto = 1.0   
    
    delta_x_optimo = None
    max_iteraciones = 100
    
    for _ in range(max_iteraciones):
        medio = (bajo + alto) / 2
        
        derivada_num = f_derivada_exacta(valor_x) 
        
        error_absoluto = abs(valor_exacto - derivada_num)
        
        if error_absoluto < epsilon:
            delta_x_optimo = medio
            bajo = medio
        else:
            alto = medio
            
        if abs(alto - bajo) < 1e-16:
            break
            
    return delta_x_optimo

delta_x_encontrado = buscar_delta_x(f, valor_x, valor_derivada_exacta, epsilon)

print(f"Derivada exacta en x={valor_x}: {valor_derivada_exacta}")
if delta_x_encontrado is not None:
    print(f"Valor de Delta x encontrado: {delta_x_encontrado:.1e}")
    print(f"Derivada numérica aproximada: {f_derivada_exacta(valor_x)}")
    print(f"Error real obtenido: {abs(valor_derivada_exacta - f_derivada_exacta(valor_x))}")
else:
    print("No se pudo encontrar un Delta x que cumpla con ese nivel de error.")