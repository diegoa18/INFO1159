#Generador de laberintos NxN.
#Crea un laberinto aleatorio con un camino garantizado entre S y M.
#S se ubica en la primera fila y M en la ultima fila.
#El resultado se guarda en un archivo CSV.

import csv
import random
import numpy as np
 
ARRIBA      = (-2,  0)
ABAJO       = ( 2,  0)
IZQUIERDA   = ( 0, -2)
DERECHA     = ( 0,  2)
DIRECCIONES = [ARRIBA, ABAJO, IZQUIERDA, DERECHA]
 
 
def generar_laberinto(n):
    laberinto  = np.ones((n, n), dtype=int)
    col_inicio = random.randint(0, n - 1)
    col_meta   = random.randint(0, n - 1)
 
    laberinto[0][col_inicio]   = 0
    laberinto[n - 1][col_meta] = 0
 
    visitados = np.zeros((n, n), dtype=bool)
    visitados[0][col_inicio] = True
    camino = [(0, col_inicio)]
 
    while camino:
        fila_actual, col_actual = camino[-1]
        vecinos_disponibles = []
 
        for df, dc in DIRECCIONES:
            fila_vecino     = fila_actual + df
            col_vecino      = col_actual  + dc
            dentro_del_mapa = 0 <= fila_vecino < n and 0 <= col_vecino < n
            if dentro_del_mapa and not visitados[fila_vecino][col_vecino]:
                vecinos_disponibles.append((fila_vecino, col_vecino))
 
        if vecinos_disponibles:
            fila_vecino, col_vecino = random.choice(vecinos_disponibles)
            fila_medio = (fila_actual + fila_vecino) // 2
            col_medio  = (col_actual  + col_vecino)  // 2
            laberinto[fila_medio][col_medio]   = 0
            laberinto[fila_vecino][col_vecino] = 0
            visitados[fila_vecino][col_vecino] = True
            camino.append((fila_vecino, col_vecino))
        else:
            camino.pop()
 
    laberinto_csv = laberinto.tolist()
    laberinto_csv[0][col_inicio]   = 'S'
    laberinto_csv[n - 1][col_meta] = 'M'
 
    return laberinto_csv, (col_inicio, 0), (col_meta, n - 1)
 
 
def guardar_csv(laberinto, ruta):
    with open(ruta, mode='w', newline='') as archivo:
        writer = csv.writer(archivo)
        for fila in laberinto:
            writer.writerow(fila)
 
 
n = int(input("Ingrese el tamaño del laberinto (NxN): "))
laberinto, inicio, meta = generar_laberinto(n)
guardar_csv(laberinto, "Laberinto/input.csv")
print(f"Laberinto {n}x{n} generado")
print(f"Inicio: {inicio} | Meta: {meta}")
 
