from __future__ import annotations

from enum import IntEnum
from typing import TypeAlias
import numpy as np
import numpy.typing as npt

class ConstraintType(IntEnum):
    LE = 1  # <=
    GE = 2  # >=
    EQ = 3  # =

class ObjectiveType(IntEnum):
    MAX = 1
    MIN = -1

EPSILON: float = 1e-10
MAX_ITERATIONS: int = 10000 # IMPLEMENTAR QUINTERO!!!!

Matriz: TypeAlias = npt.NDArray[np.float64]
Tupla: TypeAlias = np.ndarray[tuple[int, ...], np.dtype[np.intp]]

# Constantes para formateo de números
UMBRAL_NOTACION_CIENTIFICA_GRANDE: float = 1e4
UMBRAL_NOTACION_CIENTIFICA_PEQUENA: float = 1e-3
TOLERANCIA_MANTISA: float = 0.01
DECIMALES_NORMAL: int = 2
ANCHO_COLUMNA_DEFAULT: int = 7
ANCHO_COLUMNA_TABLAU: int = 12
