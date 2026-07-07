from typing import List, Sequence, Tuple

from cromosoma import MetricasCromosoma


def penalizacion_pausas(pausas_intermedias: int) -> float:
    return 10 * pausas_intermedias


def penalizacion_choques(choques: int) -> float:
    return 30 * choques


def penalizacion_bloque_giro(longitud_bloque: int) -> float:
    if longitud_bloque <= 1:
        return 0

    if longitud_bloque == 2:
        return 10

    if longitud_bloque == 3:
        return 30

    return 120 * (longitud_bloque - 3)


def penalizacion_bloques_giros(bloques_giros: Sequence[int]) -> float:
    total = 0.0

    for bloque in bloques_giros:
        total += penalizacion_bloque_giro(bloque)

    return total


def penalizacion_post_meta(acciones_post_meta: int) -> float:
    return 100 * acciones_post_meta


def penalizacion_detencion_prematura(q_prematuros: int) -> float:
    return 10 * q_prematuros


def penalizacion_invalidez(es_valido: bool) -> float:
    return 0 if es_valido else 10000


def funcion_objetivo_J(resultado: MetricasCromosoma) -> float:
    return (
        resultado.distancia_final
        + resultado.tau
        + penalizacion_pausas(resultado.pausas_intermedias)
        + penalizacion_choques(resultado.choques)
        + penalizacion_bloques_giros(resultado.bloques_giros)
        + penalizacion_post_meta(resultado.acciones_post_meta)
        + penalizacion_detencion_prematura(resultado.detencion_prematura)
        + penalizacion_invalidez(resultado.es_valido)
    )


# def fitness(resultado: MetricasCromosoma) -> float:
#    return -funcion_objetivo_J(resultado)
#
#
# def evaluar_poblacion(
#    poblacion: List[MetricasCromosoma],
# ) -> List[Tuple[MetricasCromosoma, float]]:
#
#    return [(resultado, fitness(resultado)) for resultado in poblacion]
#
#
# def seleccionar_mejor(
#    poblacion: List[MetricasCromosoma],
# ) -> Tuple[MetricasCromosoma, float]:
#
#    if not poblacion:
#        raise ValueError("La poblacion no puede estar vacia")
#
#    mejor_resultado = poblacion[0]
#    mejor_fitness = fitness(mejor_resultado)
#
#    for resultado in poblacion[1:]:
#        valor = fitness(resultado)
#
#        if valor > mejor_fitness:
#            mejor_resultado = resultado
#            mejor_fitness = valor
#
#    return mejor_resultado, mejor_fitness
