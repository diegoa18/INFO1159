import csv
import numpy as np
import sys

def cargar_laberinto(ruta_archivo):
    matriz_temporal = []
    
    # abrimos el archivo csv pa leerlo. asegurarse de estar en directorio /Laberinto
    with open(ruta_archivo, mode='r', encoding='utf-8') as archivo:
        lector_csv = csv.reader(archivo)
        for fila in lector_csv:
            # limpiamos los espacios por si el archivo viene mal ingresado
            fila_limpia = [celda.strip() for celda in fila]
            
            # validacion : solo simbolos permitidos (0, 1, 2, X)
            for celda in fila_limpia:
                if celda not in ['0', '1', '2', 'X']:
                    print('error: simbolo no permitido, solo se aceptan 0, 1, 2, X.')
                    sys.exit(1)
                    
            matriz_temporal.append(fila_limpia)
    
    # pasamos todo a una matriz de numpy en formato texto pa que no se pierdan los '0', '1', '2' y 'X'
    mapa_numpy = np.array(matriz_temporal, dtype=str)
    m, r = mapa_numpy.shape
    
    # validacion: lugares de inicio (1) y meta (2)
    inicios = np.argwhere(mapa_numpy == '1')
    metas = np.argwhere(mapa_numpy == '2')
    
    if len(inicios) != 1 or len(metas) != 1:
        print("error: el mapa tiene que tener exactamente un inicio (1) y una meta (2).")
        sys.exit(1)
      
    inicio = tuple(inicios[0])
    meta = tuple(metas[0])
    
    # validacion: muros x en todo el borde exterior)
    if not (np.all(mapa_numpy[0, :] == 'X') and np.all(mapa_numpy[m-1, :] == 'X') and 
            np.all(mapa_numpy[:, 0] == 'X') and np.all(mapa_numpy[:, r-1] == 'X')):
        print("error: revisar el muro del csv.")
        sys.exit(1)
        
    # validacion: vecindario despejado de inicio y meta
    def vecindario_interior_despejado(posicion):
        i, j = posicion
        # revisamos las 8 celdas alrededor
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                if di == 0 and dj == 0: 
                    continue # nos saltamos la celda central
                
                ni, nj = i + di, j + dj
                # solo revisamos las celdas interiores
                if 0 < ni < m-1 and 0 < nj < r-1:
                    if mapa_numpy[ni, nj] == 'X':
                        return False
        return True

    if not vecindario_interior_despejado(inicio) or not vecindario_interior_despejado(meta):
        print("error: hay muros bloqueando el inicio o la meta por dentro (vecindario de moore).")
        sys.exit(1)
    
    return mapa_numpy, inicio, meta

if __name__ == "__main__":
    # input de variables para el algoritmo genetico
    print("configuracion del algoritmo genetico")
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
# quitar las comillas para mostrar que se cargo del input
print("\nresultados de la carga")
print(f"inicio (fila, columna): {inicio} | meta: {meta}")
print(f"dimensiones del mapa: {mapa.shape[0]} filas x {mapa.shape[1]} columnas")
print("matriz cargada:")
print(mapa)
"""