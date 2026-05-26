from sympy import symbols, sympify, lambdify, sin, pi
import numpy as np
import matxplotlib.pyplot as plt

x = symbols('x')

expr = input("ingrese la expresión: ")
expresion = sympify(expr) # la convierte a expresion de sympy

fun = lambdify(x, expr, 'numpy')

xa = float(input("ingrese el valor de xa: ")) # valor de x1
xb = float(input("ingrese el valor de xb: ")) # valor de x2
landa = float(input("ingrese el valor de landa [0,1]: ")) # valor de lambda entre 0 y 1

rango = np.arange(0, 3, 0.01) # rango de x para graficar
y = fun(rango) # valores de f(x) para el rango de x

valor = np.arange(xa, xb, 0.01)  # rango de x entre xa y xb para graficar la función en ese intervalo
interp = np.linspace(xa, xb, 100) # valores de x para la interpolacion

interp_y = fun(xa) + (interp - xa) * (fun(xb) - fun(xa)) / (xb - xa) # calcula la interpolación

convexa = True
concava = True

for xa in valor:
    for xb in valor:
        rhs = landa*fun(xa) + (1-landa)*fun(xb) # lado derecho
        lhs = fun(landa*xa + (1-landa)*xb) # lado izquierdo

        if lhs <= rhs:
            concava = False
        if lhs >= rhs:
            convexa = False

if convexa:
    print("La función es convexa")
elif concava:
    print("La función es cóncava")
else:
    print("La función es lineal")

plt.plot(rango, y, label='f(x)', color='black')
plt.plot(valor, fun(valor), label='f(x)', color='blue')
plt.plot(interp, interp_y, color="red", label="Interpolación")
plt.scatter([xa, xb], [fun(xa), fun(xb)],color='red', s=40) # marca los puntos xa y xb
plt.xlabel("x")
plt.ylabel("f(x)")
plt.show()
