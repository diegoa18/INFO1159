from sympy import symbols 
import numpy as np

x1_sim, x2_sim, theta_sim = symbols('x1 x2 theta')

def funcion(x1, x2, t):
    lineal = 1.20*x1 + 1.16*x2
    cuadratico = 2*x1**2 + x2**2 + (x1 + x2)**2
    return lineal - t * cuadratico

mi_formula = funcion(x1_sim, x2_sim, theta_sim)

theta_valor = 0.00006   
delta_x = 0.1
rango = np.arange(0, 5 + delta_x, delta_x)

f_maxima = -999999    
x1_estrella = 0
x2_estrella = 0
 
for valor_1 in rango:
    for valor_2 in rango:
        if valor_1 + valor_2 <= 5:
            valor_actual = mi_formula.subs({x1_sim: valor_1, x2_sim: valor_2, theta_sim: theta_valor})

            if valor_actual > f_maxima:
                f_maxima = valor_actual 
                x1_estrella = valor_1      
                x2_estrella = valor_2

print(f"Resultado para theta = {theta_valor}")
print(f"x* = ({x1_estrella}, {x2_estrella})")
print(f"f(x*) = {f_maxima}")