import sympy as sp

def gradiente(funcion, punto, deltas):
    variables = sorted(funcion.free_symbols, key=lambda x: x.name)
    gradiente = []

    punto_original = dict(zip(variables, punto))
    f_original = funcion.subs(punto_original)

    for i in range(len(variables)):
        copia = punto.copy()
        copia[i] += deltas[i]

        punto_perturbado = dict(zip(variables, copia))
        f_perturbada = funcion.subs(punto_perturbado)
        derivada_parcial = float((f_perturbada - f_original) / deltas[i])
        gradiente.append(derivada_parcial)

    return gradiente

if __name__ == "__main__":
    funcion = sp.sympify(input("Ingrese la función: "))
    punto = list(map(float, input("Ingrese el punto (separado por espacio): ").split(" ")))
    deltas = list(map(float, input("Ingrese los delta_x (separados por espacio): ").split(" ")))

    resultado = gradiente(funcion, punto, deltas)
    print(f"Gradiente numérico: {resultado}")
