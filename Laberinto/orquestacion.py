# Orquestación algoritmo Genético para la Resolución de Laberintos
import random
import sys

import matplotlib.pyplot as plt

# Imports reales de los modulos de tus companeros
import parser_csv
from cromosoma import Cromosoma, simular
from fitness import fitness, funcion_objetivo_J
from seleccion import ordenar_poblacion, seleccionar_padres

# ---- ORQUESTACION Y RESULTADOS ----


def graficar_evolucion_j(historico_j):
    # Grafica el mejor J por generacion en escala logaritmica
    plt.figure(figsize=(8, 5))
    plt.plot(
        range(1, len(historico_j) + 1),
        historico_j,
        marker="o",
        linestyle="-",
        color="b",
    )
    plt.yscale("log")
    plt.title("Evolución del Mejor Valor de Función Objetivo (Global)")
    plt.xlabel("Generación")
    plt.ylabel("Función Objetivo J (Escala Log)")
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.show()


def graficar_proporcion_validas(historico_validas):
    # Grafica el porcentaje de soluciones validas por generacion
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(historico_validas) + 1), historico_validas, color="g")
    plt.title("Proporción de Soluciones Válidas por Generación")
    plt.xlabel("Generación")
    plt.ylabel("Proporción (0.0 a 1.0)")
    plt.ylim(-0.1, 1.1)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.show()


def mostrar_mejores_unicos(mejores_unicos, mejor_j_global, metricas_mejor):
    # Imprime en consola los cromosomas unicos empatados en el mejor J
    print("\n" + "=" * 50)
    print("REPORTE DE MEJORES CROMOSOMAS ÚNICOS")
    print("=" * 50)
    print(f"Mejor J global encontrado: {mejor_j_global:.2f}")
    print(f"Fitness (ϕ): {fitness(metricas_mejor):.2f}")

    for i, (cromo, metricas) in enumerate(mejores_unicos, 1):
        print(f"\nSolución Única #{i}")

        print(f"Cromosoma: {cromo.genes}")
        print(f"Pasos tomados (tau): {metricas.tau}")
        print(f"Trayectoria auditada (X, Y):")
        for fila, col in metricas.trayectoria:
            print(f" -> {col}, {fila}")
    print("=" * 50)


def main(simulador_fn=simular, params=None, silencioso=False):
    # 1. Solicitar parametros por consola o usar los provistos
    if params is None:
        if not silencioso:
            print("--- Configuración del Algoritmo Genético ---")
        ruta_csv, n, pm, N_pob, G, ps, seed = parser_csv.pedir_inputs()
    else:
        ruta_csv = params.get("ruta_csv", "input.csv")
        n = params["n"]
        pm = params["pm"]
        N_pob = params["N"]
        G = params["G"]
        ps = params["ps"]
        seed = params["seed"]

    if n < 1:
        print("error: n debe ser al menos 1")
        sys.exit(1)
    if not (0 < pm < 1):
        print("error: pm debe estar entre 0 y 1")
        sys.exit(1)
    if G < 1:
        print("error: G debe ser al menos 1")
        sys.exit(1)
    if not (0 < ps < 1):
        print("error: ps debe estar entre 0 y 1")
        sys.exit(1)

    if N_pob < 3:
        print("error: la poblacion (N) debe ser almenos 3")
        sys.exit(1)

    # Validacion rapida de poblacion impar (requisito de la pauta)
    if N_pob % 2 == 0:
        if not silencioso:
            print(
                "Aviso: La población debe ser impar. Sumando 1 al tamaño de la población."
            )
        N_pob += 1

    params = {"n": n, "pm": pm, "N": N_pob, "G": G, "ps": ps, "seed": seed}

    # 2. Inicializar mapa y semilla usando random.Random para aislar la semilla
    mapa, inicio, meta = parser_csv.cargar_laberinto(ruta_csv)
    rng = random.Random(params["seed"])

    # 3. Poblacion inicial usando la clase de Diego
    poblacion = [Cromosoma.aleatorio(params["n"], rng) for _ in range(params["N"])]

    mejor_global_absoluto = None
    mejor_j_global = float("inf")
    historico_j = []
    historico_validas = []

    # Set para guardar solo elementos unicos que empaten en el primer lugar
    mejores_unicos = set()

    if not silencioso:
        print("\nIniciando evolución...")

    # 4. Ciclo de generaciones
    for gen in range(params["G"]):
        # Simular y emparejar cada cromosoma con sus metricas (Tupla: Individuo)
        evaluados = [(c, simulador_fn(c, mapa, inicio, meta)) for c in poblacion]

        # Ordenar con la funcion de Joaquin (usa la funcion J de Dani internamente)
        ordenados = ordenar_poblacion(evaluados)

        # Rescatar el mejor de esta generacion (el indice 0 tras ordenar)
        mejor_actual_cromo, mejor_actual_metricas = ordenados[0]
        j_actual = funcion_objetivo_J(mejor_actual_metricas)

        # Actualizar elitismo y tracking de unicos absolutos
        if j_actual < mejor_j_global:
            mejor_j_global = j_actual
            mejor_global_absoluto = (mejor_actual_cromo, mejor_actual_metricas)
            mejores_unicos.clear()

        for cromo, metricas in evaluados:
            if funcion_objetivo_J(metricas) == mejor_j_global:
                mejores_unicos.add((cromo, metricas))

        # Guardar metricas para los graficos
        historico_j.append(mejor_j_global)
        cant_validas = sum(1 for _, m in evaluados if m.es_valido)
        historico_validas.append(cant_validas / params["N"])

        # 5. Elitismo: construir la nueva poblacion reservando el primer cupo
        nueva_pob_cromosomas = [mejor_global_absoluto[0].copiar()]

        # Rellenar los N-1 espacios restantes con hijos generados por operadores
        while len(nueva_pob_cromosomas) < params["N"]:
            p1, p2 = seleccionar_padres(ordenados, params["ps"], rng)

            punto_corte = rng.randint(1, params["n"] - 1)
            # p1[0] y p2[0] acceden al objeto Cromosoma dentro de la tupla Individuo
            h1, h2 = p1[0].cruzar_un_punto(p2[0], punto_corte)

            h1.mutar(params["pm"], rng)
            h2.mutar(params["pm"], rng)

            nueva_pob_cromosomas.append(h1)
            # Asegurar no pasarnos del limite impar al agregar el segundo hijo
            if len(nueva_pob_cromosomas) < params["N"]:
                nueva_pob_cromosomas.append(h2)

        # Reemplazar la poblacion antigua por la nueva
        poblacion = nueva_pob_cromosomas

    if not silencioso:
        print("Evolución terminada. Generando reportes...")

    # 6. Mostrar resultados exigidos por la pauta
    if not silencioso:
        mostrar_mejores_unicos(mejores_unicos, mejor_j_global, mejor_global_absoluto[1])
        graficar_evolucion_j(historico_j)
        graficar_proporcion_validas(historico_validas)


if __name__ == "__main__":
    main()
