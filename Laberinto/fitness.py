from typing import List, Sequence, Tuple

from cromosoma import MetricasCromosoma


# penalizamos al agente si mete 'Q's intermedias, porque las 'Q's solo deben ir al final tras cruzar la meta
def penalizacion_pausas(pausas_intermedias: int) -> float:
    return 10 * pausas_intermedias


# chocar contra muros (X) debe doler bastante, por eso aplicamos una penalización de 30 puntos por choque
def penalizacion_choques(choques: int) -> float:
    return 30 * choques


# giros consecutivos inútiles gastan energía y pasos. los penalizamos exponencialmente según el tamaño del bloque de giros
def penalizacion_bloque_giro(longitud_bloque: int) -> float:
    if longitud_bloque <= 1:
        return 0

    if longitud_bloque == 2:
        return 10

    if longitud_bloque == 3:
        return 30

    return 120 * (longitud_bloque - 3)


# recorremos todos los bloques de giros consecutivos acumulados en la trayectoria para penalizarlos
def penalizacion_bloques_giros(bloques_giros: Sequence[int]) -> float:
    total = 0.0

    for bloque in bloques_giros:
        total += penalizacion_bloque_giro(bloque)

    return total


# si el agente cruza la meta y sigue dando vueltas en lugar de pararse con 'Q', lo penalizamos con 100 puntos por cada acción post-meta
def penalizacion_post_meta(acciones_post_meta: int) -> float:
    return 100 * acciones_post_meta


# si el agente no es válido, penalizamos la cantidad de 'Q's al final porque significa que se detuvo antes de tiempo (detención prematura)
def penalizacion_detencion_prematura(q_prematuros: int) -> float:
    return 10 * q_prematuros


# castigo de 10,000 puntos para asesinar de inmediato la aptitud de los cromosomas inválidos (los que no terminan en Q tras la meta o nunca llegaron)
def penalizacion_invalidez(es_valido: bool) -> float:
    return 0 if es_valido else 10000


# la gran función de costo J(x). sumamos la distancia manhattan final, los pasos tomados (tau) y todas las penalizaciones acumuladas
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


# el fitness phi es el negativo de J, porque nuestro algoritmo genético maximiza y queremos minimizar el costo J(x)
def fitness(resultado: MetricasCromosoma) -> float:
    return -funcion_objetivo_J(resultado)


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
#
