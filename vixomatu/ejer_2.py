import matplotlib.pyplot as plt
import numpy as np
import sympy as sp

x = sp.symbols("x")

Δλ = np.arange(0, 1, 0.01)

funcion_str = input("ingrese la funcion: ")
funcion = sp.sympify(funcion_str)

f_num = sp.lambdify(x, funcion)

puntos_str = input("Ingrese los puntos xa y xb: ")
xa_val, xb_val = sp.sympify(puntos_str)


def convexa(funcion, var, xa, xb):
    for λ in Δλ:
        evaluado = λ * xa + (1 - λ) * xb
        lado_izq = funcion.subs(var, evaluado)
        lado_der = λ * funcion.subs(var, xa) + (1 - λ) * funcion.subs(var, xb)

        if lado_izq > lado_der:
            return False

    return True


def concava(funcion, var, xa, xb):
    for λ in Δλ:
        evaluado = λ * xa + (1 - λ) * xb
        lado_izq = funcion.subs(var, evaluado)
        lado_der = λ * funcion.subs(var, xa) + (1 - λ) * funcion.subs(var, xb)

        if lado_izq < lado_der:
            return False

    return True


def convex_o_conca(funcion, var, xa, xb):
    es_conv = convexa(funcion, var, xa, xb)
    es_conc = concava(funcion, var, xa, xb)

    if es_conv and es_conc:
        return "La función es Convexa y Cóncava"
    elif es_conv:
        return "La función es Convexa"
    elif es_conc:
        return "La función es Cóncava"
    else:
        return "La función no es ni convexa ni cóncava entre esos puntos"


resultado = convex_o_conca(funcion, x, xa_val, xb_val)
print(f"resultado: {resultado}")

xa_f = float(xa_val)
xb_f = float(xb_val)
margen = (xb_f - xa_f) * 0.5

x_vals = np.linspace(float(xa_val) - margen, float(xb_val) + margen, 100)

fig, ax = plt.subplots()
ax.plot(x_vals, f_num(x_vals))
ax.plot(x_vals, f_der(x_vals), "bo")

ax.set_xlabel("x")
ax.set_ylabel("f(x)")
ax.set_title("Gráfico de la función")

ax.axhline(0, color="black", linewidth=0.5)
ax.axvline(0, color="black", linewidth=0.5)

plt.show()


"""
xa_f = float(xa_val)
xb_f = float(xb_val)
margen = (xb_f - xa_f) * 0.5

x_vals = np.linspace(xa_f - margen, xb_f + margen, 100)

plt.figure(figsize=(8, 5))
plt.plot(x_vals, f_num(x_vals))
plt.xlabel("x")
plt.ylabel("f(x)")
plt.title("Gráfico de la función")
plt.show()
"""
