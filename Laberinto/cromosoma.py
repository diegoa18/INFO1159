import random
from typing import Iterator, List

IZQUIERDA = "IZQUIERDA"
DERECHA = "DERECHA"
ADELANTE = "ADELANTE"
QUIETO = "QUIETO"

MOVIMIENTOS = [IZQUIERDA, DERECHA, ADELANTE, QUIETO]


def cromosoma_aleatorio(longitud: int) -> "Cromosoma":
    return Cromosoma([random.choice(MOVIMIENTOS) for _ in range(longitud)])


class Cromosoma:
    def __init__(self, genes: List[str]):
        self._genes = genes

    def __getitem__(self, index):
        return self._genes[index]

    def __setitem__(self, index, valor):
        self._genes[index] = valor

    def __len__(self) -> int:
        return len(self._genes)

    def __iter__(self) -> Iterator[str]:
        return iter(self._genes)

    def __reversed__(self) -> Iterator[str]:
        return reversed(self._genes)

    def __repr__(self) -> str:
        return f"Cromosoma({self._genes})"

    @property
    def genes(self) -> List[str]:
        return self._genes

    def copiar(self) -> "Cromosoma":
        return Cromosoma(self._genes.copy())
