from __future__ import annotations

import random
from typing import Iterator, List, NamedTuple, Sequence, Tuple

__all__ = [
    "H",
    "A",
    "M",
    "Q",
    "ACCIONES",
    "DIRS",
    "HORARIO",
    "ANTIHORARIO",
    "TRANSITABLE",
    "ALTERNATIVAS",
    "Cromosoma",
    "MetricasCromosoma",
    "simular",
]

# definimos las cuatro acciones del alfabeto de control del robot:
# H = girar 90 grados sentido Horario
# A = girar 90 grados sentido Antihorario
# M = Moverse un casillero adelante en la dirección actual
# Q = Quedarse quieto
H = "H"
A = "A"
M = "M"
Q = "Q"
ACCIONES = (H, A, M, Q)

# mapeo de movimientos cardinales según el sentido del autómata
DIRS: dict[str, Tuple[int, int]] = {
    "N": (-1, 0),
    "E": (0, 1),
    "S": (1, 0),
    "O": (0, -1),
}

# rotaciones del autómata en sentido horario
HORARIO: dict[str, str] = {"N": "E", "E": "S", "S": "O", "O": "N"}

# rotaciones del autómata en sentido antihorario
ANTIHORARIO: dict[str, str] = {"N": "O", "O": "S", "S": "E", "E": "N"}

# el agente solo puede transitar sobre caminos vacíos '0', inicio '1' o meta '2'
TRANSITABLE: frozenset[str] = frozenset({"0", "1", "2"})

# alternativas para mutación: asegura que si un gen muta, cambie sí o sí a una acción distinta (no-nula)
ALTERNATIVAS: dict[str, Tuple[str, str, str]] = {
    H: (A, M, Q),
    A: (H, M, Q),
    M: (H, A, Q),
    Q: (H, A, M),
}


# esta es la estructura que almacena todas las métricas de la simulación de un cromosoma
class MetricasCromosoma(NamedTuple):
    distancia_final: int
    tau: int
    es_valido: bool
    pausas_intermedias: int
    choques: int
    bloques_giros: Tuple[int, ...]
    acciones_post_meta: int
    detencion_prematura: int
    ultima_llegada: int
    llegadas_efectivas: Tuple[int, ...]
    trayectoria: Tuple[Tuple[int, int], ...]
    posicion_final: Tuple[int, int]
    direccion_final: str


# clase que encapsula el genotipo (vector de strings) del cromosoma
class Cromosoma:
    def __init__(self, genes: List[str]) -> None:
        self._genes = genes

    # genera un individuo aleatorio rellenando el vector con el alfabeto
    @classmethod
    def aleatorio(cls, longitud: int, rng: random.Random) -> Cromosoma:
        return cls([rng.choice(ACCIONES) for _ in range(longitud)])

    def __getitem__(self, index: int) -> str:
        return self._genes[index]

    def __setitem__(self, index: int, valor: str) -> None:
        self._genes[index] = valor

    def __len__(self) -> int:
        return len(self._genes)

    def __iter__(self) -> Iterator[str]:
        return iter(self._genes)

    def __repr__(self) -> str:
        return f"Cromosoma({''.join(self._genes)})"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Cromosoma) and self._genes == other._genes

    def __hash__(self) -> int:
        return hash(tuple(self._genes))

    @property
    def genes(self) -> List[str]:
        return self._genes

    def copiar(self) -> Cromosoma:
        return Cromosoma(self._genes.copy())

    # cruce monopunto: corta los genes de los dos padres en un índice 'c' y mezcla las mitades
    def cruzar_un_punto(self, otro: Cromosoma, c: int) -> Tuple[Cromosoma, Cromosoma]:
        if not (1 <= c <= len(self) - 1):
            raise ValueError(
                f"Punto de corte c debe estar entre 1 y {len(self) - 1}, se obtuvo {c}"
            )
        return (
            Cromosoma(self._genes[:c] + otro._genes[c:]),
            Cromosoma(otro._genes[:c] + self._genes[c:]),
        )

    # mutación gen a gen: cada gen tiene una probabilidad 'pm' de mutar a otra acción diferente
    def mutar(self, pm: float, rng: random.Random) -> None:
        for i in range(len(self._genes)):
            if rng.random() < pm:
                self._genes[i] = rng.choice(ALTERNATIVAS[self._genes[i]])


# busca hacia atrás el último gen que no sea 'Q' para evaluar detenciones prematuras
def _ultimo_no_q(genes: Sequence[str], n: int) -> int:
    for i in range(n - 1, -1, -1):
        if genes[i] != Q:
            return i
    return -1


# simulador secuencial estándar en CPU: corre el autómata paso a paso y recolecta métricas de simulación
def simular(
    cromosoma: Cromosoma,
    mapa: Sequence[Sequence[str]],
    inicio: Tuple[int, int],
    meta: Tuple[int, int],
) -> MetricasCromosoma:
    n = len(cromosoma)
    filas = len(mapa)
    cols = len(mapa[0])

    p = inicio
    d = "S" # empezamos apuntando al sur según el estándar acordado
    trayectoria: List[Tuple[int, int]] = [p]
    llegadas_efectivas: List[int] = []
    bloques_giros: List[int] = []
    contador_giros = 0
    choques = 0
    ha_llegado = p == meta
    acciones_post_meta = 0

    # iteramos sobre las acciones del cromosoma ejecutando el autómata en el laberinto
    for k, gen in enumerate(cromosoma, start=1):
        prev_p = p

        if gen == H:
            d = HORARIO[d]
            contador_giros += 1
        elif gen == A:
            d = ANTIHORARIO[d]
            contador_giros += 1
        elif gen == M:
            dr, dc = DIRS[d]
            nr = p[0] + dr
            nc = p[1] + dc
            # si la celda es transitable y está dentro de los límites del mapa, nos movemos
            if 0 <= nr < filas and 0 <= nc < cols and mapa[nr][nc] in TRANSITABLE:
                p = (nr, nc)
                if contador_giros:
                    # si veníamos de girar y nos movemos, registramos la cantidad de giros acumulados
                    bloques_giros.append(contador_giros)
                    contador_giros = 0
            else:
                # si es muro o se sale de la matriz, es un choque
                choques += 1

        # acumulamos penalizaciones post-meta
        if ha_llegado and gen in (H, A, M) and prev_p == meta:
            acciones_post_meta += 1
            
        # registramos el paso en que se pisa la meta por primera vez
        if p == meta and prev_p != meta:
            llegadas_efectivas.append(k)
        if p == meta:
            ha_llegado = True

        trayectoria.append(p)

    # si quedan giros colgados al final del cromosoma, los registramos
    if contador_giros:
        bloques_giros.append(contador_giros)

    distancia_final = abs(p[0] - meta[0]) + abs(p[1] - meta[1])

    ultima_llegada = max(llegadas_efectivas) if llegadas_efectivas else 0
    es_valido = False
    tau = n + 1

    # validamos que si llegó a la meta, todos los genes restantes sean 'Q' para que la solución sea válida
    if llegadas_efectivas and ultima_llegada < n:
        if all(cromosoma[k] == Q for k in range(ultima_llegada, n)):
            es_valido = True
            tau = ultima_llegada

    # contamos cuántas 'Q's intermedias o prematuras ocurrieron en la simulación
    ultimo_no_q = _ultimo_no_q(cromosoma.genes, n)
    pausas_intermedias = 0
    if ultimo_no_q >= 0:
        for i in range(ultimo_no_q):
            if cromosoma[i] == Q:
                pausas_intermedias += 1

    detencion_prematura = 0
    if not es_valido:
        for i in range(n - 1, -1, -1):
            if cromosoma[i] == Q:
                detencion_prematura += 1
            else:
                break

    # retornamos el tuple completo con las métricas finales eager evaluadas
    return MetricasCromosoma(
        distancia_final=distancia_final,
        tau=tau,
        es_valido=es_valido,
        pausas_intermedias=pausas_intermedias,
        choques=choques,
        bloques_giros=tuple(bloques_giros),
        acciones_post_meta=acciones_post_meta,
        detencion_prematura=detencion_prematura,
        ultima_llegada=ultima_llegada,
        llegadas_efectivas=tuple(llegadas_efectivas),
        trayectoria=tuple(trayectoria),
        posicion_final=p,
        direccion_final=d,
    )
