import sympy as sp

"""en que direccion aumenta mas rapidamente f(xn) y con que intensidad"""


def gradiente_numerico(variables, func, x_vals, deltas):
    punto = {v: xv for v, xv in zip(variables, x_vals)}
    f_x = float(func.subs(punto))

    grad = []
    for var, d in zip(variables, deltas):
        punto_pert = punto.copy()
        punto_pert[var] += d
        grad.append((float(func.subs(punto_pert)) - f_x) / d)

    print(f"gradiente: {grad}")


if __name__ == "__main__":
    variables = tuple(sp.symbols(input("ingrese las variables (espacios): ")))
    func = sp.sympify(input("ingrese f(x): "))
    x_vals = list(map(float, input("ingrese el punto x (coma): ").split(",")))
    deltas = list(map(float, input("ingrese los Δx (coma): ").split(",")))

    gradiente_numerico(variables, func, x_vals, deltas)
