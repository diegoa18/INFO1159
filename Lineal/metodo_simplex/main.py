from __future__ import annotations

import sys
sys.path.insert(0, "..")

import numpy as np
from parser import pedir_problema
from simplex import InfeasibleError, UnboundedError
from simplex.simplex_solver import SimplexSolver
from simplex.problem import LinearProgram
from simplex.types import ConstraintType, ObjectiveType

# convierte formato neutro a LinearProgram (simplex)
def _convertir(tipo: str, fo: list[float], restricciones) -> LinearProgram:

    A = np.array([r[0] for r in restricciones], dtype=np.float64)
    b = np.array([r[1] for r in restricciones], dtype=np.float64)
    c = np.array(fo, dtype=np.float64)

    tipo_map = {
        "<=": ConstraintType.LE,
        ">=": ConstraintType.GE,
        "=": ConstraintType.EQ,
    }
    constraints_arr = np.array(
        [tipo_map[r[2]] for r in restricciones], dtype=np.int_
    )

    obj = ObjectiveType.MAX if tipo == "max" else ObjectiveType.MIN

    return LinearProgram(A=A, b=b, c=c, constraints=constraints_arr, objective=obj)


# pedimos problema y convertimos a formato simplex, luego resolvemos e imprimimos resultados
def main() -> None:
    tipo, fo, restricciones = pedir_problema()
    lp = _convertir(tipo, fo, restricciones)

    solver = SimplexSolver(trazo=True)

    try:
        solution = solver.solve(lp)
        print("SOLUCION OPTIMA")
        print("-" * 60)

        print(f"valor optimo: {solution.optimal_value:.6f}")

        for i, val in enumerate(solution.variables, start=1):
            print(f"x{i} = {val:.6f}")

        print(f"iteraciones: {solution.iterations}")

    except UnboundedError as e:
        print("Problema no acotado")
        print(e)

    except InfeasibleError as e:
        print("Problema infactible")
        print(e)


if __name__ == "__main__":
    main()