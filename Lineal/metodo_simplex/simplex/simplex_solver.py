from __future__ import annotations

import numpy as np
from typing import Optional

from .exceptions import StabilityError, UnboundedError
from .pivot import pivot
from .problem import LinearProgram, Solution
from .tableau import Tableau , print_tableau
from .types import ObjectiveType, EPSILON, MAX_ITERATIONS

class SimplexSolver:
    def __init__(self, trazo: bool = False):
        self.trazo = trazo

    def solve(self, problem: LinearProgram):
        tableau = Tableau.desde_programa_lineal(problem)

        es_min = problem.objective == ObjectiveType.MIN
        
        if len(tableau.rango_artificiales) > 0:
            return self.two_phases(tableau, problem, es_min)
        
        return self.standar(tableau, problem, es_min)

    def standar(self, tableau, problem, es_min):
        if es_min:
            tableau.datos[tableau.fila_objetivo, :tableau.num_variables_originales] = problem.c
        
        fila_z = tableau.fila_objetivo

        for i in range(tableau.num_restricciones):
            var_base = int(tableau.variables_basicas[i])
            coef_z = tableau.datos[fila_z, var_base]
            if abs(coef_z) > EPSILON:
                tableau.datos[fila_z, :] -= coef_z * tableau.datos[i, :]

        if self.trazo:
            print("\n--- TABLA INICIAL SIMPLEX ---")
            print_tableau(tableau)

        tableau, iterations = simplex_iterate(tableau, trazo=self.trazo)

        valor_optimo, variables = extract(tableau)

        if es_min:
            valor_optimo = -valor_optimo

        return Solution(valor_optimo, variables, True, iterations)

    # ejecuta el método simplex con dos fases
    def two_phases(self, tableau, problem, es_min):
        fila_z = tableau.fila_objetivo

        tableau.datos[fila_z, :] = 0
        for j in tableau.rango_artificiales:
            tableau.datos[fila_z, j] = 1.0

        print("\n--- Tableado inicial ---")
        print_tableau(tableau)
        for i in range(tableau.num_restricciones):
            var_base = int(tableau.variables_basicas[i])
            if var_base in tableau.rango_artificiales:
                tableau.datos[fila_z, :] -= tableau.datos[i, :]

        if self.trazo:
            print("\n--- TABLA INICIAL FASE 1 ---")
            print_tableau(tableau)

        tableau, iter1 = simplex_iterate(tableau, trazo=self.trazo)

        if abs(tableau.obtener_valor()) > EPSILON:
            raise StabilityError(f"Infactible: Z={tableau.obtener_valor():.4f}")

        tableau = tableau.eliminar_columnas_artificiales()
        tableau.restaurar_objetivo(problem.c, es_min)

        new_fila_z = tableau.fila_objetivo
        for i in range(tableau.num_restricciones):
            var_idx = int(tableau.variables_basicas[i])
            coef_z = tableau.datos[new_fila_z, var_idx]
            if abs(coef_z) > EPSILON:
                tableau.datos[new_fila_z, :] -= coef_z * tableau.datos[i, :]

        if self.trazo:
            print("\n--- TABLA INICIAL FASE 2 ---")
            print_tableau(tableau)

        tableau, iter2 = simplex_iterate(tableau, trazo=self.trazo)

        valor_optimo, variables = extract(tableau)

        if es_min:
            valor_optimo = -valor_optimo

        return Solution(valor_optimo, variables, True, iter1 + iter2)


# ve la fila de la función objetivo para encontrar la variable que entra
def quien_entra(tableau: Tableau, epsilon: float = EPSILON) -> Optional[int]:
    fila_z = tableau.datos[tableau.fila_objetivo, :-1]

    fila_z = np.where(abs(fila_z) < epsilon, 0.0, fila_z)

    candidatos = np.where(fila_z < -epsilon)[0]
    if len(candidatos) == 0:
        return None
    return int(min(candidatos))


# ve la fila de la tabla para encontrar la variable que sale
def quien_sale(
    tableau: Tableau, col_pivote: int, epsilon: float = EPSILON
) -> Optional[int]:

    LD = tableau.datos[:-1, -1]
    col = tableau.datos[:-1, col_pivote]

    min_ratio = np.inf
    fila_saliente = None

    for i in range(len(LD)):
        if col[i] > epsilon and LD[i] >= -epsilon:
            ratio = LD[i] / col[i]

            if ratio < min_ratio - epsilon or (
                abs(ratio - min_ratio) < epsilon
                and (fila_saliente is None or i < fila_saliente)
            ):
                min_ratio = ratio
                fila_saliente = i

    return fila_saliente


# identifica pivote, printea la tabla y llama a pivor para actualizar el tableau
def simplex_iterate(
    tableau: Tableau,
    epsilon: float = EPSILON,
    trazo: bool = False,
) -> tuple[Tableau, int]:

    iteration = 0
    vb_vistas = set()

    while True:
        if iteration >= MAX_ITERATIONS:
            raise StabilityError("Número máximo de iteraciones alcanzado")
    
        vb_actual = tuple(tableau.variables_basicas)
        if vb_actual in vb_vistas:
            raise StabilityError("Ciclaje detectado")
        vb_vistas.add(vb_actual)

        col = quien_entra(tableau, epsilon)

        if col is None:
            if trazo:
                print("Punto optimo alcanzado")
            return tableau, iteration

        fila = quien_sale(tableau, col, epsilon)
        if fila is None:
            raise UnboundedError(f"No acotado: columna pivote {col} sin fila válida")

        saliente = int(tableau.variables_basicas[fila])

        tableau = pivot(tableau, fila, col, epsilon)

        if trazo:
            print_tableau(
                tableau,
                iteracion=iteration + 1,
                fila_pivote=fila,
                columna_pivote=col,
                variable_saliente=saliente,
            )

        iteration += 1


# una vez que quien entra es None, extrae las soluciones básicas y devuelve el valor óptimo
def extract(tableau: Tableau):
    n = tableau.num_variables_originales
    x = np.zeros(n)

    for i in range(tableau.num_restricciones):
        var = int(tableau.variables_basicas[i])
        if var < n:
            x[var] = tableau.datos[i, tableau.columna_lado_derecho]

    return tableau.obtener_valor(), tuple(x)
