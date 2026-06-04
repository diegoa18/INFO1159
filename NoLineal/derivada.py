import sympy as sp


def d_numerica(x, func, x_val, deltax):
    d_numerica = (func.subs(x, x_val + deltax) - func.subs(x, x_val)) / deltax
    print(f"resultado: {sp.N(d_numerica)}")


def find_delta(x, func, x_val, tolerancia, divisiones=2):
    derivada = sp.diff(func, x)
    delta = sp.sqrt(tolerancia) * x_val
    d_num = float(derivada.subs(x, x_val))

    while not (
        abs(d_num - float((func.subs(x, x_val + delta) - func.subs(x, x_val)) / delta))
        < tolerancia
    ):
        delta /= divisiones
    print(f"Δx: {sp.N(abs(delta))}")


if __name__ == "__main__":
    x = sp.symbols("x")
    caso = input("1: derivada numerica; 2: encontrar delta: ")
    func = sp.sympify(input("ingrese f(x): "))


    match caso:
        case "1":        
            d_numerica(
                x, func, float(input("ingrese el punto x: ")), float(input("ingrese Δx: "))
            )
        case "2":
            find_delta(
                x,
                func,
                float(input("ingrese el punto x: ")),
                float(input("ingrese la tolerancia e: ")),
            )
        case _:
            print("caso no valido")
