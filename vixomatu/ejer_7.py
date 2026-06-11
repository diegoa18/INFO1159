import numpy as np
import sympy as sp

vars_str = input("ingrese las dos variables separadas por espacio: ")
variables = sp.symbols(vars_str.split())

funcion_str = input("ingrese la funcion bivariante: ")
funcion = sp.sympify(funcion_str)

punto_str = input("ingrese el vector x a evaluar: ")
vector_x = [float(val) for val in punto_str.split()]

tol = float(input("ingrese el valor de tolerancia: "))

x_sym = variables[0]
y_sym = variables[1]

f_xx_sym = sp.diff(funcion, x_sym, x_sym)
f_yy_sym = sp.diff(funcion, y_sym, y_sym)
f_xy_sym = sp.diff(funcion, x_sym, y_sym)

f_xx_num = sp.lambdify(variables, f_xx_sym, "numpy")
f_yy_num = sp.lambdify(variables, f_yy_sym, "numpy")
f_xy_num = sp.lambdify(variables, f_xy_sym, "numpy")

val_x = vector_x[0]
val_y = vector_x[1]

v_xx = float(f_xx_num(val_x, val_y))
v_yy = float(f_yy_num(val_x, val_y))
v_xy = float(f_xy_num(val_x, val_y))

hessiano = np.array([[v_xx, v_xy], [v_xy, v_yy]])


def evaluar_criterio_hessiano(matriz, tolerancia):
    f_xx_val = matriz[0, 0]
    determinante = np.linalg.det(matriz)

    print(f"matriz hessiana exacta:\n{matriz}")
    print(f"determinante: {determinante}")

    if determinante > tolerancia:
        if f_xx_val > tolerancia:
            print("conclusion: el punto es un minimo local")
        elif f_xx_val < -tolerancia:
            print("conclusion: el punto es un maximo local")
        else:
            print("conclusion: no hay informacion suficiente")
    elif determinante < -tolerancia:
        print("conclusion: el punto es un punto de silla")
    else:
        print("conclusion: no hay informacion suficiente")


evaluar_criterio_hessiano(hessiano, tol)
