from __future__ import annotations

import numpy as np

from .exceptions import PivotError
from .tableau import Tableau

# SIEMPRE SE BUSCA [[0],[0],[1],[0],[O]], osea un solo 1 en la columna
from .types import EPSILON


# pivotea la fila pivot_row usando la columna pivot_col como pivote
def pivot(
    tableau: Tableau,  # estado actual
    pivot_row: int,  # fil var saliente
    pivot_col: int,  # col var entrante
    epsilon: float = EPSILON,
) -> Tableau:
    total_vars = (
        tableau.num_variables_originales
        + tableau.num_holguras
        + tableau.num_excesos
        + tableau.num_artificiales
    )

    data = tableau.datos.copy()

    pivot_val = float(data[pivot_row, pivot_col])

    if abs(pivot_val) < epsilon:  # pivot != 0
        raise PivotError(f"Pivot value at ({pivot_row}, {pivot_col}) is zero")

    data[pivot_row] /= pivot_val  # dividir fila

    # eliminar colpivot
    for i in range(data.shape[0]):
        if i == pivot_row:  # no modificar filpivot
            continue

        # coeficiente a pitiar
        factor = float(data[i, pivot_col])
        if abs(factor) > epsilon:
            data[i] -= (
                factor * data[pivot_row]
            )  # Ri = Ri - factor * Rpivot, para colpivot = 0

    data[np.abs(data) < epsilon] = 0.0

    # actualizar VB
    new_basic = tableau.variables_basicas.copy()
    new_basic[pivot_row] = pivot_col  # ej: s1->x1

    all_vars = np.arange(total_vars, dtype=np.intp)  # para [0,1,2,..,n]

    nonbasic = np.setdiff1d(all_vars, new_basic, assume_unique=True)  # todas - VB


    nuevo_tableau = Tableau(  # nuevo objeto tableau actualizado
        datos=data,
        variables_basicas=new_basic,
        variables_no_basicas=nonbasic,
        num_variables_originales=tableau.num_variables_originales,
        num_restricciones=tableau.num_restricciones,
        num_holguras=tableau.num_holguras,
        num_excesos=tableau.num_excesos,
        num_artificiales=tableau.num_artificiales,
        objective=tableau.objective,
    )
    
    return nuevo_tableau
