import random as ra
from typing import List, Tuple

from cromosoma import Cromosoma
from fitness import ResultadoEjecucion, funcion_objetivo_J

Individuo = Tuple[Cromosoma, ResultadoEjecucion]


def prioridad_factibilidad(individuo: Individuo) -> int:
    _, resultado = individuo
    if resultado.es_valido:
        return 0
    if resultado.llego_meta:
        return 1
    return 2


def clave_ordenamiento(individuo: Individuo) -> Tuple[int, float, float, float]:
    _, resultado = individuo
    rho = prioridad_factibilidad(individuo)
    J = funcion_objetivo_J(resultado)
    D = resultado.distancia
    tau = resultado.tau
    return (rho, J, D, tau)


def ordenar_poblacion(poblacion: List[Individuo]) -> List[Individuo]:
    return sorted(poblacion, key=clave_ordenamiento)


def validar_poblacion_impar(N: int) -> bool:
    return N >= 3 and N % 2 == 1


def pesos_ranking_geometrico(N: int, ps: float) -> List[float]:
    # para un parametro Ps E (0,1) los pesos no normalizados serán
    return [ps * (1 - ps) ** (i - 1) for i in range(1, N + 1)]


def probabilidades_normalizadas(N: int, ps: float) -> List[float]:
    pesos = pesos_ranking_geometrico(N, ps)
    suma = sum(pesos)
    return [p / suma for p in pesos]


def distribucion_acumulada(N: int, ps: float) -> List[float]:
    return [(1 - (1 - ps) ** i) / (1 - (1 - ps) ** N) for i in range(1, N + 1)]


def seleccionar_parental(poblacion_ord: List[Individuo], ps: float) -> Individuo:
    N = len(poblacion_ord)
    C = distribucion_acumulada(N, ps)
    u = ra.random()
    for i, ci in enumerate(C):
        if u <= ci:
            return poblacion_ord[i]
    return poblacion_ord[-1]


def seleccionar_padres(
    poblacion_ord: List[Individuo], ps: float
) -> Tuple[Individuo, Individuo]:
    return (
        seleccionar_parental(poblacion_ord, ps),
        seleccionar_parental(poblacion_ord, ps),
    )


def elitismo(
    mejor_global: Individuo, poblacion_ord: List[Individuo]
) -> List[Individuo]:
    poblacion = poblacion_ord[:]
    poblacion[-1] = mejor_global
    return ordenar_poblacion(poblacion)


def _pruebas():
    import random as _ra

    pob: List[Individuo] = []
    for _ in range(5):
        c = Cromosoma.aleatorio(10, _ra.Random(42))
        r = ResultadoEjecucion(
            distancia=float(ra.randint(0, 10)),
            tau=float(ra.randint(1, 10)),
            es_valido=ra.choice([True, False]),
            llego_meta=ra.choice([True, False]),
            choques=ra.randint(0, 5),
            pausas_intermedias=ra.randint(0, 3),
            bloques_giros=[],
            acciones_post_meta=0,
            q_prematuros=0,
        )
        pob.append((c, r))

    print("=== Población original ===")
    for c, r in pob:
        print(
            f"  valido={r.es_valido}, llego={r.llego_meta}, D={r.distancia}, tau={r.tau}, J={funcion_objetivo_J(r)}"
        )

    print(f"\nPoblación impar (N=5): {validar_poblacion_impar(5)}")
    print(f"Población impar (N=4): {validar_poblacion_impar(4)}")

    ord_pob = ordenar_poblacion(pob)
    print("\n=== Población ordenada (mejor → peor) ===")
    for c, r in ord_pob:
        rho = prioridad_factibilidad((c, r))
        print(f"  rho={rho}, J={funcion_objetivo_J(r)}, D={r.distancia}, tau={r.tau}")

    print(
        f"\nProb normalizadas (ps=0.2): {[round(p, 4) for p in probabilidades_normalizadas(5, 0.2)]}"
    )
    print(
        f"Distribución acumulada:    {[round(c, 4) for c in distribucion_acumulada(5, 0.2)]}"
    )

    p1, p2 = seleccionar_padres(ord_pob, 0.2)
    print(
        f"\nPadres seleccionados: J={funcion_objetivo_J(p1[1])} y J={funcion_objetivo_J(p2[1])}"
    )

    mejor = ord_pob[0]
    nueva_pob = elitismo(mejor, ord_pob)
    print(f"\nElitismo: mejor reinsertado -> {len(nueva_pob)} individuos")
    print(
        f"Mejor en nueva pob: rho={prioridad_factibilidad(nueva_pob[0])}, J={funcion_objetivo_J(nueva_pob[0][1])}"
    )


if __name__ == "__main__":
    _pruebas()
