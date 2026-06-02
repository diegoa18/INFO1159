import sympy as sp

x = sp.symbols('x')

def derivada_num(funcion, punto_x, delta_x ):
    return (funcion(punto_x + delta_x) - funcion(punto_x)) / delta_x

def encontrar_deltax(funcion, f,  punto_x, epsilon, divisiones):
    derivada = sp.diff(funcion, x)
    delta_x = float(sp.sqrt(epsilon) * punto_x)
    d_num = derivada_num(f, punto_x, delta_x)

    while not (abs(float(derivada.subs(x, punto_x)) - d_num) < epsilon):
        delta_x /= divisiones
        d_num = derivada_num(f, punto_x, delta_x)

    return abs(delta_x)

if __name__ == "__main__":
    caso = input("1: derivada numerica; 2: encontrar delta: ")
    funcion = sp.sympify(input("ingrese f(x): "))
    f = sp.lambdify(x, funcion, 'numpy')

    if caso == "1":
        punto_x = float(input("ingrese el punto x: "))
        delta_x = float(input("ingrese delta_x: "))
        print(derivada_num(f, punto_x, delta_x))
    elif caso == "2":
        punto_x = float(input("ingrese el punto x: "))
        epsilon = float(input("ingrese la tolerancia e: "))
        divisiones = 2
        print(encontrar_deltax(funcion, f,  punto_x, epsilon, divisiones))
    else:
        print("caso no valido")