import sys
sys.path.insert(0, "..")

from parser import pedir_problema
from programacion_lineal import graficar, resolver_PL


def main():
    print("=== Solver Gráfico (2 variables) ===\n")

    tipo, fo, restricciones_raw = pedir_problema(modo="grafico")

    # Convertir formato neutro a formato método gráfico
    # restricciones_raw: list[(coeff, lado_derecho, tipo)]
    restricciones = []
    for coeff, lado_derecho, signo in restricciones_raw:
        restricciones.append((coeff[0], coeff[1], lado_derecho, signo))

    # Restricciones de no negatividad
    restricciones += [
        (1, 0, 0, ">="),
        (0, 1, 0, ">="),
    ]

    a, b = fo[0], fo[1]
    solucion = resolver_PL(restricciones, a, b, tipo)

    print("\nEstado:", solucion["estado"])

    if solucion["optimo"]:
        (x, y), z = solucion["optimo"]
        print(f"Óptimo en ({x}, {y}) → Z = {z}")

    graficar(solucion, restricciones)


main()