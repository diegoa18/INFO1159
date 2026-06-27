import numpy as np
import random
import csv
import matplotlib.pyplot as plt

IZQUIERDA = 0
DERECHA   = 1
ADELANTE  = 2
QUIETO    = 3

LONGITUD_CROMOSOMA = 100
TAMANO_POBLACION   = 60
GENERACIONES       = 150
TASA_MUTACION      = 0.05
P_RANKING          = 0.3


def cargar_laberinto(ruta_archivo):
    matriz_temporal = []
    inicio = None
    meta   = None
    with open(ruta_archivo, mode='r') as archivo:
        lector_csv = csv.reader(archivo)
        for y, fila in enumerate(lector_csv):
            fila_limpia = []
            for x, celda in enumerate(fila):
                if celda == 'S':
                    inicio = (x, y)
                    fila_limpia.append(0)
                elif celda == 'M':
                    meta = (x, y)
                    fila_limpia.append(0)
                else:
                    fila_limpia.append(int(celda))
            matriz_temporal.append(fila_limpia)
    return np.array(matriz_temporal), inicio, meta


MAPA, INICIO, META = cargar_laberinto("input.csv")


def crear_cromosoma():
    return [random.randint(0, 3) for _ in range(LONGITUD_CROMOSOMA)]

def crear_poblacion():
    return [crear_cromosoma() for _ in range(TAMANO_POBLACION)]


def simular(cromosoma):
    filas, cols = MAPA.shape
    x, y = INICIO
    direccion = 1
    pasos = 0

    dx = [ 0, 1,  0, -1]
    dy = [-1, 0,  1,  0]

    for accion in cromosoma:
        if accion == IZQUIERDA:
            direccion = (direccion - 1) % 4
        elif accion == DERECHA:
            direccion = (direccion + 1) % 4
        elif accion == ADELANTE:
            nx = x + dx[direccion]
            ny = y + dy[direccion]
            dentro_del_mapa = 0 <= nx < cols and 0 <= ny < filas
            if dentro_del_mapa and MAPA[ny][nx] == 0:
                x, y = nx, ny
                pasos += 1
        if (x, y) == META:
            return (x, y), pasos, True

    return (x, y), pasos, False


def fitness(cromosoma):
    pos_final, pasos, llego = simular(cromosoma)
    if llego:
        return 1.0 + 1.0 / (pasos + 1)
    distancia = abs(pos_final[0] - META[0]) + abs(pos_final[1] - META[1])
    return 1.0 / (distancia + 1)


def calcular_probabilidades(fitnesses):
    n = len(fitnesses)
    orden = np.argsort(fitnesses)[::-1]
    probs = np.zeros(n)
    for rango, idx in enumerate(orden):
        probs[idx] = P_RANKING * ((1 - P_RANKING) ** rango)
    return probs / probs.sum()

def seleccionar_padre(poblacion, probs):
    idx = np.random.choice(len(poblacion), p=probs)
    return poblacion[idx]


def cruzar(padre1, padre2):
    punto = random.randint(1, LONGITUD_CROMOSOMA - 1)
    hijo1 = padre1[:punto] + padre2[punto:]
    hijo2 = padre2[:punto] + padre1[punto:]
    return hijo1, hijo2


def mutar(cromosoma):
    return [
        random.randint(0, 3) if random.random() < TASA_MUTACION else gen
        for gen in cromosoma
    ]


def aplicar_elitismo(poblacion, fitnesses):
    idx_elite = fitnesses.index(max(fitnesses))
    return poblacion[idx_elite]


def ejecutar_ag():
    poblacion = crear_poblacion()

    hist_mejor     = []
    hist_promedio  = []
    hist_prob_meta = []

    for gen in range(GENERACIONES):
        fitnesses = [fitness(ind) for ind in poblacion]

        hist_mejor.append(max(fitnesses))
        hist_promedio.append(sum(fitnesses) / len(fitnesses))
        hist_prob_meta.append(sum(1 for f in fitnesses if f > 1.0) / TAMANO_POBLACION)

        elite = aplicar_elitismo(poblacion, fitnesses)
        probs = calcular_probabilidades(fitnesses)

        nueva_poblacion = [elite]

        while len(nueva_poblacion) < TAMANO_POBLACION:
            padre1 = seleccionar_padre(poblacion, probs)
            padre2 = seleccionar_padre(poblacion, probs)
            hijo1, hijo2 = cruzar(padre1, padre2)
            nueva_poblacion.append(mutar(hijo1))
            if len(nueva_poblacion) < TAMANO_POBLACION:
                nueva_poblacion.append(mutar(hijo2))

        poblacion = nueva_poblacion

        if hist_prob_meta[-1] >= 0.9:
            print(f"Convergencia en generacion {gen}")
            break

    fitnesses_finales = [fitness(ind) for ind in poblacion]
    mejor = poblacion[fitnesses_finales.index(max(fitnesses_finales))]
    return mejor, hist_mejor, hist_promedio, hist_prob_meta


def graficar(hist_mejor, hist_promedio, hist_prob_meta):
    gens = list(range(len(hist_mejor)))
    prob_pct = [p * 100 for p in hist_prob_meta]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Algoritmo Genetico — Laberinto", fontsize=13)

    axes[0].plot(gens, hist_mejor,    label='Mejor individuo',   color='#2ecc71', linewidth=2)
    axes[0].plot(gens, hist_promedio, label='Promedio poblacion', color='#3498db', linewidth=1.5, linestyle='--')
    axes[0].set_xlabel('Generacion')
    axes[0].set_ylabel('Fitness')
    axes[0].set_title('Evolucion del fitness')
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(gens, prob_pct, color='#8e44ad', linewidth=2)
    axes[1].fill_between(gens, prob_pct, alpha=0.15, color='#8e44ad')
    axes[1].set_xlabel('Generacion')
    axes[1].set_ylabel('% que llegan a la meta')
    axes[1].set_title('Probabilidad de llegar a la meta por generacion')
    axes[1].set_ylim(0, 100)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    mejor, hist_mejor, hist_promedio, hist_prob_meta = ejecutar_ag()

    pos_final, pasos, llego = simular(mejor)
    if llego:
        print(f"Llego a la meta en {pasos} pasos")
    else:
        distancia = abs(pos_final[0] - META[0]) + abs(pos_final[1] - META[1])
        print(f"No llego. Distancia Manhattan a la meta: {distancia}")

    graficar(hist_mejor, hist_promedio, hist_prob_meta)