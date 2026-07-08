import random
from functools import lru_cache
from typing import List, Tuple

from cromosoma import Cromosoma, MetricasCromosoma
from fitness import funcion_objetivo_J

Individuo = Tuple[Cromosoma, MetricasCromosoma]


# aquí definimos la jerarquía de factibilidad del gran deb:
# 0 -> la solución es válida (llegó y completó con 'Q's)
# 1 -> llegó a la meta pero tiene basura después (acciones no-Q post-meta)
# 2 -> ni siquiera fue capaz de encontrar la meta
def prioridad_factibilidad(individuo: Individuo) -> int:
    _, resultado = individuo
    if resultado.es_valido:
        return 0
    if len(resultado.llegadas_efectivas) > 0:
        return 1
    return 2


# esta es la clave que usa sorted() para ordenar los individuos
# ordenamos primero por si es factible/llegó (rho), luego por su valor J, luego por distancia y al final por tiempo de llegada (tau)
# esto implementa las reglas de deb al pie de la letra
def clave_ordenamiento(individuo: Individuo) -> Tuple[int, float, int, int]:
    _, resultado = individuo
    rho = prioridad_factibilidad(individuo)
    J = funcion_objetivo_J(resultado)
    D = resultado.distancia_final
    tau = resultado.tau
    return (rho, J, D, tau)


# ordenamos la población de mejor a peor usando la clave de deb
def ordenar_poblacion(poblacion: List[Individuo]) -> List[Individuo]:
    return sorted(poblacion, key=clave_ordenamiento)


# aquí calculamos el peso no normalizado de cada puesto en el ranking según la presión selectiva ps
def pesos_ranking_geometrico(N: int, ps: float) -> List[float]:
    # para un parametro Ps E (0,1) los pesos no normalizados serán
    return [ps * (1 - ps) ** (i - 1) for i in range(1, N + 1)]


# normalizamos las probabilidades para que sumen 1.0 exacto
def probabilidades_normalizadas(N: int, ps: float) -> List[float]:
    pesos = pesos_ranking_geometrico(N, ps)
    suma = sum(pesos)
    return [p / suma for p in pesos]


# ojo aquí: usamos lru_cache para que la cpu no explote recalculando esta sumatoria de selección en cada intento de cada generación.
# dado que N y ps no cambian, se calcula exactamente una sola vez y luego se lee al instante de la ram
@lru_cache(maxsize=16)
def distribucion_acumulada(N: int, ps: float) -> List[float]:
    if ps <= 0 or ps >= 1:
        raise ValueError("ps debe estar entre 0 y 1")
    probs = probabilidades_normalizadas(N, ps)
    C = []
    acumulado = 0.0
    for p in probs:
        acumulado += p
        C.append(acumulado)
    return C


# seleccionamos un padre usando la ruleta acumulativa sobre la distribución geométrica
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


# seleccionamos dos padres de forma independiente para cruzarlos
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
