import time
import warnings
import numpy as np
from numba import cuda
from numba.core.errors import NumbaPerformanceWarning
from typing import Tuple, List

from cromosoma import Cromosoma, MetricasCromosoma, simular

# silenciar los warnings de rendimiento de numba (por ejemplo, cuando el lote es chico)
warnings.simplefilter("ignore", category=NumbaPerformanceWarning)

# mapeos estaticos al reves para las direcciones cardinales
DIR_MAP_REV = {0: "N", 1: "E", 2: "S", 3: "O"}


@cuda.jit
def _kernel_simular_cuda(
    poblacion_num: np.ndarray,
    mapa_num: np.ndarray,
    inicio_r: int,
    inicio_c: int,
    meta_r: int,
    meta_c: int,
    N: int,
    n: int,
    filas: int,
    cols: int,
    distancia_final_arr: np.ndarray,
    tau_arr: np.ndarray,
    es_valido_arr: np.ndarray,
    pausas_intermedias_arr: np.ndarray,
    choques_arr: np.ndarray,
    acciones_post_meta_arr: np.ndarray,
    detencion_prematura_arr: np.ndarray,
    ultima_llegada_arr: np.ndarray,
    p_r_final_arr: np.ndarray,
    p_c_final_arr: np.ndarray,
    d_final_arr: np.ndarray,
    num_llegadas_arr: np.ndarray,
    llegadas_efectivas_arr: np.ndarray,
    num_bloques_arr: np.ndarray,
    bloques_giros_arr: np.ndarray,
    trayectorias_arr: np.ndarray
):
    # este es el kernel de cuda, cada hilo se va a encargar de simular la trayectoria de un individuo de la poblacion
    i = cuda.grid(1)
    if i < N:
        p_r = inicio_r
        p_c = inicio_c
        d = 2  # la 'S' empieza apuntando al sur (mapeado como 2)

        trayectorias_arr[i, 0, 0] = p_r
        trayectorias_arr[i, 0, 1] = p_c

        num_llegadas = 0
        num_bloques = 0
        contador_giros = 0
        choques = 0
        ha_llegado = (p_r == meta_r and p_c == meta_c)
        acciones_post_meta = 0

        # simulamos paso a paso los movimientos en el laberinto
        for k in range(1, n + 1):
            gen = poblacion_num[i, k - 1]
            prev_p_r = p_r
            prev_p_c = p_c

            if gen == 0:  # H (sentido horario)
                d = (d + 1) % 4
                contador_giros += 1
            elif gen == 1:  # A (sentido antihorario)
                d = (d - 1) % 4
                contador_giros += 1
            elif gen == 2:  # M (moverse hacia adelante)
                dr = 0
                dc = 0
                if d == 0:    # norte
                    dr = -1
                elif d == 1:  # este
                    dc = 1
                elif d == 2:  # sur
                    dr = 1
                elif d == 3:  # oeste
                    dc = -1

                nr = p_r + dr
                nc = p_c + dc

                # vemos si podemos pasar (valores menores a 3 son validos, la X es 3)
                if 0 <= nr < filas and 0 <= nc < cols and mapa_num[nr, nc] < 3:
                    p_r = nr
                    p_c = nc
                    if contador_giros > 0:
                        bloques_giros_arr[i, num_bloques] = contador_giros
                        num_bloques += 1
                        contador_giros = 0
                else:
                    choques += 1

            if ha_llegado and (gen == 0 or gen == 1 or gen == 2) and (prev_p_r == meta_r and prev_p_c == meta_c):
                acciones_post_meta += 1
            if (p_r == meta_r and p_c == meta_c) and (prev_p_r != meta_r or prev_p_c != meta_c):
                llegadas_efectivas_arr[i, num_llegadas] = k
                num_llegadas += 1
            if p_r == meta_r and p_c == meta_c:
                ha_llegado = True

            trayectorias_arr[i, k, 0] = p_r
            trayectorias_arr[i, k, 1] = p_c

        if contador_giros > 0:
            bloques_giros_arr[i, num_bloques] = contador_giros
            num_bloques += 1

        distancia_final = abs(p_r - meta_r) + abs(p_c - meta_c)

        ultima_llegada = 0
        if num_llegadas > 0:
            ultima_llegada = llegadas_efectivas_arr[i, num_llegadas - 1]

        es_valido = False
        tau = n + 1
        if num_llegadas > 0 and ultima_llegada < n:
            todos_q = True
            for idx in range(ultima_llegada, n):
                if poblacion_num[i, idx] != 3:  # la Q es 3
                    todos_q = False
                    break
            if todos_q:
                es_valido = True
                tau = ultima_llegada

        ultimo_no_q = -1
        for idx in range(n - 1, -1, -1):
            if poblacion_num[i, idx] != 3:
                ultimo_no_q = idx
                break

        pausas_intermedias = 0
        if ultimo_no_q >= 0:
            for idx in range(ultimo_no_q):
                if poblacion_num[i, idx] == 3:
                    pausas_intermedias += 1

        detencion_prematura = 0
        if not es_valido:
            for idx in range(n - 1, -1, -1):
                if poblacion_num[i, idx] == 3:
                    detencion_prematura += 1
                else:
                    break

        distancia_final_arr[i] = distancia_final
        tau_arr[i] = tau
        es_valido_arr[i] = es_valido
        pausas_intermedias_arr[i] = pausas_intermedias
        choques_arr[i] = choques
        acciones_post_meta_arr[i] = acciones_post_meta
        detencion_prematura_arr[i] = detencion_prematura
        ultima_llegada_arr[i] = ultima_llegada
        p_r_final_arr[i] = p_r
        p_c_final_arr[i] = p_c
        d_final_arr[i] = d
        num_llegadas_arr[i] = num_llegadas
        num_bloques_arr[i] = num_bloques


# cache para no repetir la conversion de mapas
_cache_mapas = {}

#retorna la matriz del laberinto convertida a números y la guarda en caché para no repetir el cálculo
def obtener_mapa_num(mapa: np.ndarray) -> np.ndarray:
    mapa_id = id(mapa)
    if mapa_id in _cache_mapas:
        return _cache_mapas[mapa_id]
        
    if mapa.dtype.kind in ('U', 'S'):
        mapa_dict = {'0': 0, '1': 1, '2': 2, 'X': 3}
        mapa_num = np.ascontiguousarray(np.vectorize(mapa_dict.get)(mapa), dtype=np.int8)
    else:
        mapa_num = np.ascontiguousarray(mapa, dtype=np.int8)
        
    _cache_mapas[mapa_id] = mapa_num
    return mapa_num


def simular_poblacion_acelerada(
    poblacion: List[Cromosoma],
    mapa: np.ndarray,
    inicio: Tuple[int, int],
    meta: Tuple[int, int],
) -> List[MetricasCromosoma]:
    
    #el wrapper principal que se encarga de mover los datos de la ram a la gpu, lanzar el kernel y traer los resultados de vuelta
    N = len(poblacion)
    n = len(poblacion[0])
    filas, cols = mapa.shape
    inicio_r, inicio_c = inicio
    meta_r, meta_c = meta
    
    # 1. pasamos los genes a una matriz numerica usando una conversion vectorizada
    genes_list = [c.genes for c in poblacion]
    arr_bytes = np.array(genes_list, dtype='S1')
    
    lookup = np.zeros(256, dtype=np.int8)
    lookup[72] = 0 # 'H'
    lookup[65] = 1 # 'A'
    lookup[77] = 2 # 'M'
    lookup[81] = 3 # 'Q'
    poblacion_num = np.ascontiguousarray(lookup[arr_bytes.view(np.uint8)], dtype=np.int8)
    
    # 2. obtenemos el mapa numerico en formato contiguo en memoria C
    mapa_num = np.ascontiguousarray(obtener_mapa_num(mapa), dtype=np.int8)
    
    # 3. preparamos en memoria ram (host) los arrays que necesitan valores centinela (-1)
    llegadas_efectivas_host = np.ascontiguousarray(np.full((N, n), -1, dtype=np.int32), dtype=np.int32)
    bloques_giros_host = np.ascontiguousarray(np.full((N, n), -1, dtype=np.int32), dtype=np.int32)
    
    # 4. mandamos los datos directo a la memoria global de la gpu
    d_poblacion = cuda.to_device(poblacion_num)
    d_mapa = cuda.to_device(mapa_num)
    d_llegadas_efectivas = cuda.to_device(llegadas_efectivas_host)
    d_bloques_giros = cuda.to_device(bloques_giros_host)
    
    # reservamos espacio para los datos que la gpu nos va a responder
    d_distancia_final = cuda.device_array(N, dtype=np.int32)
    d_tau = cuda.device_array(N, dtype=np.int32)
    d_es_valido = cuda.device_array(N, dtype=np.bool_)
    d_pausas_intermedias = cuda.device_array(N, dtype=np.int32)
    d_choques = cuda.device_array(N, dtype=np.int32)
    d_acciones_post_meta = cuda.device_array(N, dtype=np.int32)
    d_detencion_prematura = cuda.device_array(N, dtype=np.int32)
    d_ultima_llegada = cuda.device_array(N, dtype=np.int32)
    d_p_r_final = cuda.device_array(N, dtype=np.int16)
    d_p_c_final = cuda.device_array(N, dtype=np.int16)
    d_d_final = cuda.device_array(N, dtype=np.int8)
    d_num_llegadas = cuda.device_array(N, dtype=np.int32)
    d_num_bloques = cuda.device_array(N, dtype=np.int32)
    d_trayectorias = cuda.device_array((N, n + 1, 2), dtype=np.int16)
    
    # 5. configuramos la cantidad de bloques e hilos cuda que vamos a usar
    threadsperblock = 256
    blockspergrid = (N + (threadsperblock - 1)) // threadsperblock
    
    # 6. lanzamos el kernel a correr en paralelo
    _kernel_simular_cuda[blockspergrid, threadsperblock](
        d_poblacion,
        d_mapa,
        inicio_r, inicio_c,
        meta_r, meta_c,
        N, n,
        filas, cols,
        d_distancia_final,
        d_tau,
        d_es_valido,
        d_pausas_intermedias,
        d_choques,
        d_acciones_post_meta,
        d_detencion_prematura,
        d_ultima_llegada,
        d_p_r_final,
        d_p_c_final,
        d_d_final,
        d_num_llegadas,
        d_llegadas_efectivas,
        d_num_bloques,
        d_bloques_giros,
        d_trayectorias
    )
    
    # esperamos a que la gpu termine de calcular todo
    cuda.synchronize()
    
    # 7. nos traemos los resultados calculados por la gpu de vuelta a la ram (host)
    dist_list = d_distancia_final.copy_to_host().tolist()
    tau_list = d_tau.copy_to_host().tolist()
    valido_list = d_es_valido.copy_to_host().tolist()
    pausas_list = d_pausas_intermedias.copy_to_host().tolist()
    choques_list = d_choques.copy_to_host().tolist()
    post_meta_list = d_acciones_post_meta.copy_to_host().tolist()
    detencion_list = d_detencion_prematura.copy_to_host().tolist()
    ult_llegada_list = d_ultima_llegada.copy_to_host().tolist()
    pr_list = d_p_r_final.copy_to_host().tolist()
    pc_list = d_p_c_final.copy_to_host().tolist()
    df_list = d_d_final.copy_to_host().tolist()
    
    num_llegadas_list = d_num_llegadas.copy_to_host().tolist()
    llegadas_list = d_llegadas_efectivas.copy_to_host().tolist()
    num_bloques_list = d_num_bloques.copy_to_host().tolist()
    bloques_list = d_bloques_giros.copy_to_host().tolist()
    trayectorias_list = d_trayectorias.copy_to_host().tolist()
    
    # 8. armamos los objetos metricascromosoma con los resultados que nos trajo la gpu
    resultados = []
    for i in range(N):
        nl = num_llegadas_list[i]
        llegadas_efectivas_tuple = tuple(llegadas_list[i][:nl])
        
        nb = num_bloques_list[i]
        bloques_giros_tuple = tuple(bloques_list[i][:nb])
        
        trayectoria_tuple = tuple(map(tuple, trayectorias_list[i]))
        posicion_final = (pr_list[i], pc_list[i])
        direccion_final = DIR_MAP_REV[df_list[i]]
        
        m = MetricasCromosoma(
            distancia_final=dist_list[i],
            tau=tau_list[i],
            es_valido=valido_list[i],
            pausas_intermedias=pausas_list[i],
            choques=choques_list[i],
            bloques_giros=bloques_giros_tuple,
            acciones_post_meta=post_meta_list[i],
            detencion_prematura=detencion_list[i],
            ultima_llegada=ult_llegada_list[i],
            llegadas_efectivas=llegadas_efectivas_tuple,
            trayectoria=trayectoria_tuple,
            posicion_final=posicion_final,
            direccion_final=direccion_final
        )
        resultados.append(m)
        
    return resultados


def simular_acelerado(
    cromosoma: Cromosoma,
    mapa: np.ndarray,
    inicio: Tuple[int, int],
    meta: Tuple[int, int],
) -> MetricasCromosoma:
    
    #simula un solo cromosoma en la gpu pasándolo como un lote de tamaño 1
    return simular_poblacion_acelerada([cromosoma], mapa, inicio, meta)[0]

    #compara el resultado de la cpu contra el de la gpu para estar seguros de que la matemática da exactamente igual
def test_equivalencia(cromosoma: Cromosoma, mapa: np.ndarray, inicio: Tuple[int, int], meta: Tuple[int, int]) -> bool:
    m_sec = simular(cromosoma, mapa, inicio, meta)
    m_acc = simular_acelerado(cromosoma, mapa, inicio, meta)
    
    fields = [
        "distancia_final", "tau", "es_valido", "pausas_intermedias", 
        "choques", "bloques_giros", "acciones_post_meta", 
        "detencion_prematura", "ultima_llegada", "llegadas_efectivas", 
        "trayectoria", "posicion_final", "direccion_final"
    ]
    
    for field in fields:
        val_sec = getattr(m_sec, field)
        val_acc = getattr(m_acc, field)
        if val_sec != val_acc:
            print(f"discrepancia en {field}: cpu={val_sec} vs gpu={val_acc}")
            return False
            
    return True


def correr_benchmark_completo(
    mapa: np.ndarray,
    inicio: Tuple[int, int],
    meta: Tuple[int, int],
    n: int,
    N: int,
    G: int,
    pm: float,
    ps: float,
    seed: int
) -> Tuple[float, float, float]:
    
    #corre la evolución completa tanto en cpu secuencial como en gpu cuda para ver qué tanta diferencia de tiempo hay"""
    import random
    from seleccion import ordenar_poblacion, seleccionar_padres
    
    # corremos una simulacion rapida para forzar la compilacion cuda previa y evitar el overhead inicial
    c_test = Cromosoma.aleatorio(n, random.Random(42))
    simular_poblacion_acelerada([c_test], mapa, inicio, meta)
    
    # -------------------------------------------------------------
    # 1. medimos la ejecucion secuencial en cpu
    # -------------------------------------------------------------
    print("corriendo el ciclo evolutivo secuencial en la cpu...")
    rng_sec = random.Random(seed)
    poblacion_sec = [Cromosoma.aleatorio(n, rng_sec) for _ in range(N)]
    
    t_inicio_sec = time.perf_counter()
    for gen in range(G):
        evaluados = [(c, simular(c, mapa, inicio, meta)) for c in poblacion_sec]
        ordenados = ordenar_poblacion(evaluados)
        mejor_actual = ordenados[0]
        nueva_pob = [mejor_actual[0].copiar()]
        
        while len(nueva_pob) < N:
            p1, p2 = seleccionar_padres(ordenados, ps, rng_sec)
            punto_corte = rng_sec.randint(1, n - 1)
            h1, h2 = p1[0].cruzar_un_punto(p2[0], punto_corte)
            h1.mutar(pm, rng_sec)
            h2.mutar(pm, rng_sec)
            nueva_pob.append(h1)
            if len(nueva_pob) < N:
                nueva_pob.append(h2)
        poblacion_sec = nueva_pob
    t_fin_sec = time.perf_counter()
    t_sec = t_fin_sec - t_inicio_sec
    
    # -------------------------------------------------------------
    # 2. medimos la ejecucion acelerada en gpu nvidia cuda
    # -------------------------------------------------------------
    print("corriendo el ciclo evolutivo acelerado en la gpu...")
    rng_acc = random.Random(seed)
    poblacion_acc = [Cromosoma.aleatorio(n, rng_acc) for _ in range(N)]
    
    t_inicio_acc = time.perf_counter()
    for gen in range(G):
        metricas_lote = simular_poblacion_acelerada(poblacion_acc, mapa, inicio, meta)
        evaluados = list(zip(poblacion_acc, metricas_lote))
        ordenados = ordenar_poblacion(evaluados)
        mejor_actual = ordenados[0]
        nueva_pob = [mejor_actual[0].copiar()]
        
        while len(nueva_pob) < N:
            p1, p2 = seleccionar_padres(ordenados, ps, rng_acc)
            punto_corte = rng_acc.randint(1, n - 1)
            h1, h2 = p1[0].cruzar_un_punto(p2[0], punto_corte)
            h1.mutar(pm, rng_acc)
            h2.mutar(pm, rng_acc)
            nueva_pob.append(h1)
            if len(nueva_pob) < N:
                nueva_pob.append(h2)
        poblacion_acc = nueva_pob
    t_fin_acc = time.perf_counter()
    t_acc = t_fin_acc - t_inicio_acc
    
    speedup = t_sec / t_acc
    
    # reporte en minusculas con los resultados finales del benchmark
    print("\n" + "="*60)
    print("      reporte comparativo de rendimiento (cpu vs gpu)")
    print("="*60)
    print(f"parámetros del algoritmo genético:")
    print(f"  - tamaño de la población (n):  {N}")
    print(f"  - longitud del cromosoma (n):  {n}")
    print(f"  - número de generaciones (g):  {G}")
    print(f"  - probabilidad de mutación (pm): {pm:.4f}")
    print(f"  - presión selectiva (ps):      {ps:.4f}")
    print(f"  - semilla aleatoria (seed):    {seed}")
    print(f"  - dimensión del laberinto:     {mapa.shape[0]}x{mapa.shape[1]}")
    print("-"*60)
    print(f"tiempos de ejecución de la evolución completa:")
    print(f"  - tiempo en cpu secuencial (t_sec):             {t_sec:.5f} s")
    print(f"  - tiempo en gpu cuda (t_acc):                  {t_acc:.5f} s")
    print(f"    -> factor de aceleración neto (s):            {speedup:.2f}x")
    print("="*60 + "\n")
    
    return t_sec, t_acc, speedup
