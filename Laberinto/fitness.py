from dataclasses import dataclass
from typing import List, Sequence, Tuple

@dataclass
class ResultadoEjecucion:
    """Resumen completo de una ejecución de cromosoma"""
    distancia: float
    tau: float
    es_valido: bool
    llego_meta: bool
    choques: int
    pausas_intermedias: int
    bloques_giros: List[int]
    acciones_post_meta: int
    q_prematuros: int


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


def funcion_objetivo_J(resultado: ResultadoEjecucion) -> float:
    """Calcula J(x)  segun la pauta del proyecto"""

    return (
        resultado.distancia
        + resultado.tau
        + penalizacion_pausas(resultado.pausas_intermedias)
        + penalizacion_choques(resultado.choques)
        + penalizacion_bloques_giros(resultado.bloques_giros)
        + penalizacion_post_meta(resultado.acciones_post_meta)
        + penalizacion_detencion_prematura(resultado.q_prematuros)
        + penalizacion_invalidez(resultado.es_valido)
    )


def fitness(resultado: ResultadoEjecucion) -> float:
    return -funcion_objetivo_J(resultado)


def evaluar_poblacion(
    poblacion: List[ResultadoEjecucion],
) -> List[Tuple[ResultadoEjecucion, float]]:
    """Calcula el fitness de cada ResultadoEjecucion de la población."""

    return [(resultado, fitness(resultado)) for resultado in poblacion]


def seleccionar_mejor(
    poblacion: List[ResultadoEjecucion],
) -> Tuple[ResultadoEjecucion, float]:

    if not poblacion:
        raise ValueError("La poblacion no puede estar vacia")

    mejor_resultado = poblacion[0]
    mejor_fitness = fitness(mejor_resultado)

    for resultado in poblacion[1:]:
        valor = fitness(resultado)

        if valor > mejor_fitness:
            mejor_resultado = resultado
            mejor_fitness = valor

    return mejor_resultado, mejor_fitness


def _mostrar_resultado_prueba(nombre: str, resultado: ResultadoEjecucion) -> None:
    """Imprime J, fitness y todas las penalizaciones"""

    print(f"{nombre}")
    print(f"ResultadoEjecucion: {resultado}")
    print(f"D(x): {resultado.distancia}")
    print(f"tau(x): {resultado.tau}")
    print(f"PQ(x): {penalizacion_pausas(resultado.pausas_intermedias)}")
    print(f"PC(x): {penalizacion_choques(resultado.choques)}")
    print(f"PR(x): {penalizacion_bloques_giros(resultado.bloques_giros)}")
    print(f"PA(x): {penalizacion_post_meta(resultado.acciones_post_meta)}")
    print(
        f"Pprem(x): {penalizacion_detencion_prematura(resultado.q_prematuros)}"
    )
    print(f"Pinv(x): {penalizacion_invalidez(resultado.es_valido)}")
    print(f"J(x): {funcion_objetivo_J(resultado)}")
    print(f"fitness: {fitness(resultado)}")
    print("-" * 60)


def ejecutar_pruebas() -> None:
    """Casos de prueba hardcodeados para verificar el módulo manualmente."""

    casos = [
        (
            "Solución válida",
            ResultadoEjecucion(
                distancia=12,
                tau=4,
                es_valido=True,
                llego_meta=True,
                choques=0,
                pausas_intermedias=1,
                bloques_giros=[1, 2, 3],
                acciones_post_meta=0,
                q_prematuros=0,
            ),
        ),
        (
            "Solución inválida",
            ResultadoEjecucion(
                distancia=8,
                tau=3,
                es_valido=False,
                llego_meta=False,
                choques=2,
                pausas_intermedias=0,
                bloques_giros=[4],
                acciones_post_meta=1,
                q_prematuros=1,
            ),
        ),
        (
            "Muchos choques",
            ResultadoEjecucion(
                distancia=10,
                tau=2,
                es_valido=True,
                llego_meta=True,
                choques=7,
                pausas_intermedias=2,
                bloques_giros=[2],
                acciones_post_meta=0,
                q_prematuros=0,
            ),
        ),
        (
            "Muchos giros",
            ResultadoEjecucion(
                distancia=6,
                tau=1,
                es_valido=True,
                llego_meta=True,
                choques=0,
                pausas_intermedias=0,
                bloques_giros=[1, 2, 3, 4, 5],
                acciones_post_meta=0,
                q_prematuros=0,
            ),
        ),
        (
            "Muchas pausas",
            ResultadoEjecucion(
                distancia=5,
                tau=1,
                es_valido=True,
                llego_meta=True,
                choques=0,
                pausas_intermedias=9,
                bloques_giros=[],
                acciones_post_meta=0,
                q_prematuros=0,
            ),
        ),
        (
            "Detención prematura",
            ResultadoEjecucion(
                distancia=15,
                tau=6,
                es_valido=True,
                llego_meta=False,
                choques=1,
                pausas_intermedias=1,
                bloques_giros=[2, 2],
                acciones_post_meta=0,
                q_prematuros=4,
            ),
        ),
    ]

    print("PRUEBAS DEL MODULO FITNESS")
    print("=" * 60)

    for nombre, resultado in casos:
        _mostrar_resultado_prueba(nombre, resultado)

    mejor, valor = seleccionar_mejor([resultado for _, resultado in casos])
    print("MEJOR INDIVIDUO DE LA POBLACION DE PRUEBA")
    print(mejor)
    print(f"fitness: {valor}")


if __name__ == "__main__":
    ejecutar_pruebas()