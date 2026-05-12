import numpy as np

# añadir sympy (lo quite pa optimizar y no exploque mi pc :c)
# da los mismos resultados que el del diego, pero el suyo más rapido, no se por qué xd

teta = float(input("Ingrese el valor de teta: "))
salto = float(input("Ingrese el valor del salto: "))
dx = np.arange(0, 5000+salto, salto) 

tupla = []

#recorre todos los puntos posibles
for x1 in dx:
    for x2 in dx:
        if x1 + x2 <= 5000: #restricción
            valor = (1.20*x1 + 1.16*x2 - teta*(2*x1**2 + x2**2 + (x1+x2)**2))
            tupla.append(([x1, x2], valor)) # añade el punto y su valor

valores = [t[1] for t in tupla] # extrae los valores de la función            
indice = np.argmax(valores)
resultado = tupla[indice]

print(f"x* = ({resultado[0][0]}, {resultado[0][1]})")
print(f"f(x*) = {resultado[1]}")