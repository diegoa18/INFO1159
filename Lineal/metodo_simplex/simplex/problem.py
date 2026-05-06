from __future__ import annotations

from dataclasses import dataclass
from typing import Optional  # Optional[algo] -> type o None

import numpy as np

from .exceptions import InvalidLPError
from .types import Tupla, ConstraintType, Matriz, ObjectiveType, Tupla

# representacion formal LP, antess de ir al solver pues
# RECORDAR QUE LP => max/min (c^T)*x, sujeto a: Ax (<=,>=,=) b; x >= 0
@dataclass(frozen=True)
class LinearProgram:
    A: Matriz # restricciones
    b: Matriz # lado derecho
    c: Matriz # coeficientes  
    constraints: Tupla  # tipo desigualdad
    objective: ObjectiveType  # MAX/MIN

    def __init__(
        self,
        A: Matriz,
        b: Matriz,
        c: Matriz,
        constraints: Optional[Tupla] = None,
        objective: ObjectiveType = ObjectiveType.MAX,  # max default
    ) -> None:
        if constraints is None:
            constraints = np.full(
                A.shape[0], ConstraintType.LE, dtype=np.int_
            )  # no definicion -> Ax <= b

        m, n = A.shape  # m = restricciones, n = variables
        # VALIDACIONES DIMENCIONALES m X n
        if b.shape != (m,):
            raise InvalidLPError(f"b debe tener la forma ({m},), tiene {b.shape}")

        if c.shape != (n,):
            raise InvalidLPError(f"c debe de tener la forma ({n},), tiene {c.shape}")

        if constraints.shape != (m,):
            raise InvalidLPError(
                f"restricciones debe tener la forma ({m},), tiene {constraints.shape}"
            )

        if (  # evitar NaN y +-Inf
            not np.all(np.isfinite(A))
            or not np.all(np.isfinite(b))
            or not np.all(np.isfinite(c))
        ):
            raise InvalidLPError("todos los elementos deben ser finitos")

        valid = {  # simbolo valido en la restriccion
            int(ConstraintType.LE),
            int(ConstraintType.GE),
            int(ConstraintType.EQ),
        }
        if not all(int(ct) in valid for ct in constraints):
            raise InvalidLPError("tipo de restriccion invalida")

        # solo al inicializar, bypass a frozen para asignar atributos
        object.__setattr__(self, "A", A)
        object.__setattr__(self, "b", b)
        object.__setattr__(self, "c", c)
        object.__setattr__(self, "constraints", constraints)
        object.__setattr__(self, "objective", objective)

    # PROPIEDADES M X N (NUMERO RESTRICCIONES O VARIABLES)
    @property
    def m(self) -> int:
        return self.A.shape[0]

    @property
    def n(self) -> int:
        return self.A.shape[1]

@dataclass(frozen=True)
class Solution:
    optimal_value: float
    variables: tuple[float, ...]  # vector solucion
    is_optimal: bool
    iterations: int