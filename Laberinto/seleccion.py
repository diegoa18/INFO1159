import random
from typing import List, Tuple

from cromosoma import Cromosoma, MetricasCromosoma
from fitness import funcion_objetivo_J

Individuo = Tuple[Cromosoma, MetricasCromosoma]


def prioridad_factibilidad(individuo: Individuo) -> int:
    _, resultado = individuo
    if resultado.es_valido:
        return 0
    if len(resultado.llegadas_efectivas) > 0:
        return 1
    return 2


def clave_ordenamiento(individuo: Individuo) -> Tuple[int, float, int, int]:
    _, resultado = individuo
    rho = prioridad_factibilidad(individuo)
    J = funcion_objetivo_J(resultado)
    D = resultado.distancia_final
    tau = resultado.tau
    return (rho, J, D, tau)


def ordenar_poblacion(poblacion: List[Individuo]) -> List[Individuo]:
    return sorted(poblacion, key=clave_ordenamiento)


# def pesos_ranking_geometrico(N: int, ps: float) -> List[float]:
#    # para un parametro Ps E (0,1) los pesos no normalizados serán
#    return [ps * (1 - ps) ** (i - 1) for i in range(1, N + 1)]
#
#
# def probabilidades_normalizadas(N: int, ps: float) -> List[float]:
#    pesos = pesos_ranking_geometrico(N, ps)
#    suma = sum(pesos)
#    return [p / suma for p in pesos]


def distribucion_acumulada(N: int, ps: float) -> List[float]:
    if ps <= 0 or ps >= 1:
        raise ValueError("ps debe estar entre 0 y 1")
    return [(1 - (1 - ps) ** i) / (1 - (1 - ps) ** N) for i in range(1, N + 1)]


def seleccionar_parental(
    poblacion_ord: List[Individuo], ps: float, rng: random.Random
) -> Individuo:
    N = len(poblacion_ord)
    C = distribucion_acumulada(N, ps)
    u = rng.random()
    for i, ci in enumerate(C):
        if u <= ci:
            return poblacion_ord[i]
    return poblacion_ord[-1]


def seleccionar_padres(
    poblacion_ord: List[Individuo], ps: float, rng: random.Random
) -> Tuple[Individuo, Individuo]:
    return (
        seleccionar_parental(poblacion_ord, ps, rng),
        seleccionar_parental(poblacion_ord, ps, rng),
    )


# def elitismo(
#    mejor_global: Individuo, poblacion_ord: List[Individuo]
# ) -> List[Individuo]:
#    return [mejor_global]
