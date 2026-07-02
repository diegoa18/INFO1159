import csv
import numpy as np

def cargar_laberinto(ruta_archivo):
    matriz_temporal = []
    
    # abrimos el archivo csv pa leerlo ASEGURARSE DE ESTAR EN DIRECTORIO /Laberinto
    with open(ruta_archivo, mode='r', encoding='utf-8') as archivo:
        lector_csv = csv.reader(archivo)
        for fila in lector_csv:
            # limpiamos los espacios por si el archivo viene mal ingresado
            fila_limpia = [celda.strip() for celda in fila]
            matriz_temporal.append(fila_limpia)
    
    # pasamos todo a una matriz de numpy en formato texto pa que no se pierdan los '0', '1', '2' y 'X'
    mapa_numpy = np.array(matriz_temporal, dtype=str)
    
    # pillamos las coordenadas de donde esta el inicio (1) y la meta (2)
    pos_inicio = np.argwhere(mapa_numpy == '1')[0]
    pos_meta = np.argwhere(mapa_numpy == '2')[0]
    
    inicio = tuple(pos_inicio)
    meta = tuple(pos_meta)
    
    return mapa_numpy, inicio, meta


# input de variables para el algoritmo genetico
print("=== configuracion del algoritmo genetico ===")
ruta_csv = input("ingresa la ruta del archivo csv (ej: input.csv): ")

# los datos obligatorios que pide el profe en el pdf
n = int(input("longitud del cromosoma (n): "))
pm = float(input("probabilidad de mutacion (pm, ej: 0.1): "))
N_pob = int(input("tamano de la poblacion (n, debe ser impar): "))
G = int(input("numero de generaciones (g): "))
ps = float(input("presion selectiva (ps, ej: 0.05): "))
seed = int(input("semilla aleatoria (seed): "))


mapa, inicio, meta = cargar_laberinto(ruta_csv)

"""
#QUITAR LAS COMILLAS PARA MOSTRAR QUE SE CARGO DEL INPUT
print("\n=== resultados de la carga ===")
print(f"inicio (fila, columna): {inicio} | meta: {meta}")
print(f"dimensiones del mapa: {mapa.shape[0]} filas x {mapa.shape[1]} columnas")
print("matriz cargada:")
print(mapa)
"""