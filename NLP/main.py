import numpy as np

teta = float(input("Ingrese el valor de teta: "))
dx = 1
rango = np.arange(0, 5000 + dx, dx) 

tupla = []

#recorre todos los puntos posibles
for x1 in rango:
    for x2 in rango:
        if x1 + x2 <= 5000: #restricción
            valor = (1.20*x1 + 1.16*x2 - teta*(2*x1**2 + x2**2 + (x1+x2)**2))
            tupla.append(([x1, x2], valor)) # añade el punto y su valor

valores = [t[1] for t in tupla] # extrae los valores de la función            
indice = np.argmax(valores)
resultado = tupla[indice]

print(f"x* = ({resultado[0][0]}, {resultado[0][1]})")
print(f"f(x*) = {resultado[1]}")