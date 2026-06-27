import csv
import numpy as np

# cargamos el archivo csv, y repasamos sus datos para encontrar la celda de salida y llegada
# pasamos esa celda a valor 0 y guardamos la posicion de esta, luego lo transformamos a un array de numpy
def cargar_laberinto_numpy(ruta_archivo):
    matriz_temporal = []
    inicio = None
    meta = None

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

    # convertimos la lista a un arreglo de numpy
    mapa_numpy = np.array(matriz_temporal)

    return mapa_numpy, inicio, meta

# --- PRUEBA ---
mapa, inicio, meta = cargar_laberinto_numpy("Laberinto/input.csv")

print(f"Inicio: {inicio} | Meta: {meta}")
print("Matriz NumPy (Solo 0s y 1s):")
print(mapa)