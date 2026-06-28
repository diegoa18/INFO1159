"""
Por ahora los cromosomas están hardcodeados.
Más adelante serán generados por el algoritmo genético.
Cada cromosoma corresponde a una lista de movimientos.
"""
def fitness(cromosoma):
    """
    El fitness corresponde a la cantidad de
    instrucciones 'QUIETO' consecutivas
    desde el final hacia el inicio(lo q dijo el diego)
    """
    contador = 0

    for movimiento in reversed(cromosoma):

        if movimiento == "QUIETO":
            contador += 1
        else:
            break

    return contador


def evaluar_poblacion(poblacion):
    """
    Calcula el fitness de cada cromosoma.
    """
    resultados = []

    for cromosoma in poblacion:
        resultados.append((cromosoma, fitness(cromosoma)))

    return resultados


def seleccionar_mejor(poblacion):
    """
    Devuelve cromosoma con mayor fitness.
    Si existen empates, devuelve el primero.
    """

    mejor = None
    mejor_fitness = -1

    for cromosoma in poblacion:

        valor = fitness(cromosoma)

        if valor > mejor_fitness:
            mejor = cromosoma
            mejor_fitness = valor

    return mejor, mejor_fitness


# CROMOSOMAS HARDCODEADOS (SOLO PRUEBAS)

cromosoma1 = [
    "IZQUIERDA",
    "ADELANTE",
    "QUIETO",
    "QUIETO",
    "QUIETO"
]

cromosoma2 = [
    "DERECHA",
    "QUIETO",
    "ADELANTE",
    "QUIETO",
    "QUIETO"
]

cromosoma3 = [
    "ADELANTE",
    "DERECHA",
    "IZQUIERDA",
    "QUIETO"
]

cromosoma4 = [
    "ADELANTE",
    "DERECHA",
    "IZQUIERDA",
    "ADELANTE"
]

cromosoma5 = [
    "QUIETO",
    "QUIETO",
    "QUIETO",
    "QUIETO",
    "QUIETO"
]


# POBLACIÓN (HARDCODEADA)

poblacion = [
    cromosoma1,
    cromosoma2,
    cromosoma3,
    cromosoma4,
    cromosoma5
]


# =========================================
# PRUEBAS
# =========================================

print("FITNESS DE CADA CROMOSOMA\n")

resultados = evaluar_poblacion(poblacion)

for i, (cromosoma, valor) in enumerate(resultados, start=1):
    print(f"Cromosoma {i}")
    print(cromosoma)
    print(f"Fitness: {valor}")
    print("-" * 40)

mejor, valor = seleccionar_mejor(poblacion)

print("\nMEJOR CROMOSOMA")
print(mejor)
print("Fitness:", valor)


# ===========================================================
# FUTURO DEL PROYECTO
# ===========================================================

# Cuando exista el algoritmo genético ya NO habrá
# cromosomas hardcodeados.
#
# La población será generada automáticamente, por ejemplo:
#
# poblacion = generar_poblacion(tamano_poblacion,
#                               longitud_cromosoma)
#
# Luego el código será exactamente el mismo:
#
# resultados = evaluar_poblacion(poblacion)
#
# mejor, fitness_mejor = seleccionar_mejor(poblacion)
#
# La función fitness NO tendrá que modificarse.