import numpy as np
from sympy import symbols, sympify

x, y = symbols('x y')
teta = 0.01 # salto de valores
dx = np.arange(0, 5, teta) 

tupla = []
expr = sympify(1.20*x + 1.16*y - teta*(2*x**2 + y**2 + (x+y)**2))

#recorre todos los puntos posibles
for x1 in dx:
    for x2 in dx:
        if x1 + x2 <= 5: #restricción
            valor = float(expr.subs({x: x1, y: x2})) #subs, sustituye los valores x e y
            tupla.append(([x1, x2], valor)) # añade el punto y su valor

valores = [t[1] for t in tupla] # extrae los valores de la función            
indice = np.argmax(valores)
resultado = tupla[indice]

print(f"x* = ({resultado[0][0]}, {resultado[0][1]})")
print(f"f(x*) = {resultado[1]}")