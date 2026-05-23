import sympy as sp


def def_derivate():
    x = sp.symbols("x")
    func = sp.sympify(input("ingrese f(x): "))
    x_val, deltax = float(input("ingrese el punto x: ")), float(input("ingrese Δx: "))
    d_numerica = (func.subs(x, x_val + deltax) - func.subs(x, x_val)) / deltax
    print(f"resultado: {sp.N(d_numerica)}")


def find_delta():
    eps_maq = 2.22e-16
    x = sp.symbols("x")
    func = sp.sympify(input("ingrese f(x): "))
    x_val, tolerancia = (
        float(input("ingrese el punto x: ")),
        float(input("ingrese la tolerancia e: ")),
    )
    derivada = sp.diff(func, x)
    fp = float(derivada.subs(x, x_val))

    delta = sp.sqrt(eps_maq) * max(abs(x_val), 1.0)

    while (
        abs(fp - float((func.subs(x, x_val + delta) - func.subs(x, x_val)) / delta))
        >= tolerancia
    ):
        delta /= 2

    print(f"Δx: {sp.N(delta)}")
