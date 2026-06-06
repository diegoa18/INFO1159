import numpy as np
import sympy as sp


def hessiana_numerica(variables, func, x_vals, deltas):
    punto = dict(zip(variables, x_vals))

    def evaluar(cambios):
        p = punto.copy()
        for var, delta in cambios.items():
            p[var] += delta
        return float(func.subs(p))

    f_x = evaluar({})
    n = len(variables)
    H = np.zeros((n, n))

    for i in range(n):
        var, d = variables[i], deltas[i]
        H[i, i] = (evaluar({var: d}) - 2 * f_x + evaluar({var: -d})) / d**2

        for j in range(i + 1, n):
            vj, dj = variables[j], deltas[j]
            val = (
                evaluar({var: d, vj: dj})
                - evaluar({var: d, vj: -dj})
                - evaluar({var: -d, vj: dj})
                + evaluar({var: -d, vj: -dj})
            ) / (4 * d * dj)
            H[i, j] = H[j, i] = val

    return H
