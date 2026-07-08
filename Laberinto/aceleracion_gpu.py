import time
import warnings
import inspect
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
    # este es el kernel de cuda: cada hilo físico de la gpu simula a un individuo de la población
    # calculamos el índice global del hilo en la grilla de ejecución unidimensional
    i = cuda.grid(1)
    if i < N:
        # inicializamos la posición del robot en el laberinto
        p_r = inicio_r
        p_c = inicio_c
        d = 2  # la 'S' empieza apuntando al sur (mapeado como 2 para operar con operaciones modulares)

        # la trayectoria en la posición inicial se guarda en la primera posición del array de salida de la gpu
        trayectorias_arr[i, 0, 0] = p_r
        trayectorias_arr[i, 0, 1] = p_c

        num_llegadas = 0
        num_bloques = 0
        contador_giros = 0
        choques = 0
        ha_llegado = (p_r == meta_r and p_c == meta_c)
        acciones_post_meta = 0

        # simulamos paso a paso en la gpu cada movimiento codificado en los genes del cromosoma
        for k in range(1, n + 1):
            gen = poblacion_num[i, k - 1]
            prev_p_r = p_r
            prev_p_c = p_c

            if gen == 0:  # H (sentido horario: norte -> este -> sur -> oeste)
                d = (d + 1) % 4
                contador_giros += 1
            elif gen == 1:  # A (sentido antihorario: norte -> oeste -> sur -> este)
                d = (d - 1) % 4
                contador_giros += 1
            elif gen == 2:  # M (moverse hacia adelante según el frente del autómata)
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

                # validamos colisión: si está dentro del laberinto y es transitable (< 3, o sea, 0, 1 o 2 en el mapa)
                if 0 <= nr < filas and 0 <= nc < cols and mapa_num[nr, nc] < 3:
                    p_r = nr
                    p_c = nc
                    # si veníamos acumulando giros y finalmente nos movimos, cerramos y registramos el bloque de giros
                    if contador_giros > 0:
                        bloques_giros_arr[i, num_bloques] = contador_giros
                        num_bloques += 1
                        contador_giros = 0
                else:
                    # choco contra un muro exterior/interior o se intentó salir del laberinto
                    choques += 1

            # registramos penalización post-meta si sigue moviéndose después de haber pisado la meta
            if ha_llegado and (gen == 0 or gen == 1 or gen == 2) and (prev_p_r == meta_r and prev_p_c == meta_c):
                acciones_post_meta += 1
            # guardamos el paso de llegada si cruza la meta por primera vez
            if (p_r == meta_r and p_c == meta_c) and (prev_p_r != meta_r or prev_p_c != meta_c):
                llegadas_efectivas_arr[i, num_llegadas] = k
                num_llegadas += 1
            if p_r == meta_r and p_c == meta_c:
                ha_llegado = True

            # registramos las coordenadas en el array tridimensional de trayectorias
            trayectorias_arr[i, k, 0] = p_r
            trayectorias_arr[i, k, 1] = p_c

        # si el cromosoma terminó y quedaron giros sin cerrar, los guardamos en el bloque final
        if contador_giros > 0:
            bloques_giros_arr[i, num_bloques] = contador_giros
            num_bloques += 1

        # cálculo de la distancia manhattan al objetivo al final del recorrido
        distancia_final = abs(p_r - meta_r) + abs(p_c - meta_c)

        ultima_llegada = 0
        if num_llegadas > 0:
            # recuperamos el paso de la última llegada efectiva a la meta
            ultima_llegada = llegadas_efectivas_arr[i, num_llegadas - 1]

        # validación de factibilidad (es_valido y tau):
        # si pisó la meta, comprobamos que el resto de las acciones hasta el final sean únicamente 'Q's (3)
        es_valido = False
        tau = n + 1
        if num_llegadas > 0 and ultima_llegada < n:
            todos_q = True
            for idx in range(ultima_llegada, n):
                if poblacion_num[i, idx] != 3:  # la Q es 3 en el lookup
                    todos_q = False
                    break
            if todos_q:
                es_valido = True
                tau = ultima_llegada

        # buscamos la última acción que no haya sido una 'Q'
        ultimo_no_q = -1
        for idx in range(n - 1, -1, -1):
            if poblacion_num[i, idx] != 3:
                ultimo_no_q = idx
                break

        # calculamos pausas intermedias: apariciones de 'Q' antes de la última acción no-Q
        pausas_intermedias = 0
        if ultimo_no_q >= 0:
            for idx in range(ultimo_no_q):
                if poblacion_num[i, idx] == 3:
                    pausas_intermedias += 1

        # calculamos detención prematura: cantidad de 'Q's al final de la simulación en individuos inválidos
        detencion_prematura = 0
        if not es_valido:
            for idx in range(n - 1, -1, -1):
                if poblacion_num[i, idx] == 3:
                    detencion_prematura += 1
                else:
                    break

        # guardamos todas las métricas en los respectivos arrays de salida en la VRAM de la GPU
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


# caché en ram para no repetir la traducción del mapa de caracteres a números np.int8
_cache_mapas = {}
# caché en vram: subimos la matriz del mapa del laberinto a la memoria de la gpu una sola vez y guardamos el device pointer aquí
_cache_d_mapa = {}
# pool de arrays de cuda: pre-asignamos y reciclamos los arreglos de salida en la gpu para no pagar la costosa creación de memoria de cuda en cada generación
_device_arrays_cache = {}
# caché del interceptor de población: almacena las métricas calculadas en el lote para que las n-1 llamadas secuenciales se sirvan al instante
_poblacion_cache = {}

# lookup table estática para traducir los caracteres 'H', 'A', 'M', 'Q' a enteros {0, 1, 2, 3} de forma vectorizada usando indexación directa
LOOKUP_GENES = np.zeros(256, dtype=np.int8)
LOOKUP_GENES[72] = 0  # ASCII de 'H'
LOOKUP_GENES[65] = 1  # ASCII de 'A'
LOOKUP_GENES[77] = 2  # ASCII de 'M'
LOOKUP_GENES[81] = 3  # ASCII de 'Q'


# traduce el mapa del laberinto a números np.int8 y lo guarda en caché local de la cpu
def obtener_mapa_num(mapa: np.ndarray) -> np.ndarray:
    mapa_id = id(mapa)
    if mapa_id in _cache_mapas:
        return _cache_mapas[mapa_id]
        
    if mapa.dtype.kind in ('U', 'S'):
        # convertimos caracteres '0', '1', '2', 'X' a valores 0, 1, 2, 3 en memoria contigua en C
        mapa_dict = {'0': 0, '1': 1, '2': 2, 'X': 3}
        mapa_num = np.ascontiguousarray(np.vectorize(mapa_dict.get)(mapa), dtype=np.int8)
    else:
        mapa_num = np.ascontiguousarray(mapa, dtype=np.int8)
        
    _cache_mapas[mapa_id] = mapa_num
    return mapa_num


# pre-asigna u obtiene del caché todos los arreglos de memoria CUDA requeridos para lanzar el kernel.
# esto evita sobrecargar el driver gráfico de nvidia instanciando y liberando memoria en cada generación
def obtener_device_arrays(N: int, n: int):
    key = (N, n)
    if key in _device_arrays_cache:
        return _device_arrays_cache[key]
        
    arrays = {
        "d_poblacion": cuda.device_array((N, n), dtype=np.int8),
        "d_llegadas_efectivas": cuda.device_array((N, n), dtype=np.int32),
        "d_bloques_giros": cuda.device_array((N, n), dtype=np.int32),
        "d_distancia_final": cuda.device_array(N, dtype=np.int32),
        "d_tau": cuda.device_array(N, dtype=np.int32),
        "d_es_valido": cuda.device_array(N, dtype=np.bool_),
        "d_pausas_intermedias": cuda.device_array(N, dtype=np.int32),
        "d_choques": cuda.device_array(N, dtype=np.int32),
        "d_acciones_post_meta": cuda.device_array(N, dtype=np.int32),
        "d_detencion_prematura": cuda.device_array(N, dtype=np.int32),
        "d_ultima_llegada": cuda.device_array(N, dtype=np.int32),
        "d_p_r_final": cuda.device_array(N, dtype=np.int16),
        "d_p_c_final": cuda.device_array(N, dtype=np.int16),
        "d_d_final": cuda.device_array(N, dtype=np.int8),
        "d_num_llegadas": cuda.device_array(N, dtype=np.int32),
        "d_num_bloques": cuda.device_array(N, dtype=np.int32),
        "d_trayectorias": cuda.device_array((N, n + 1, 2), dtype=np.int16),
    }
    _device_arrays_cache[key] = arrays
    return arrays


# esta subclase de MetricasCromosoma evita construir miles de tuplas/listas pesadas en RAM (como la trayectoria o giros)
# durante cada paso de la evolución. Solo si la CPU accede explícitamente a estas propiedades (ej: al final de la corrida
# para graficar o reportar la solución), decodificamos el array plano traído de la GPU a tuplas de Python.
class LazyMetricasCromosoma(MetricasCromosoma):
    def __new__(
        cls,
        distancia_final,
        tau,
        es_valido,
        pausas_intermedias,
        choques,
        bloques_giros_raw,
        nb,
        acciones_post_meta,
        detencion_prematura,
        ultima_llegada,
        llegadas_efectivas_raw,
        nl,
        trayectoria_raw,
        posicion_final,
        direccion_final
    ):
        # instanciamos la clase base inicializando las tuplas perezosas como None
        obj = super().__new__(
            cls,
            distancia_final,
            tau,
            es_valido,
            pausas_intermedias,
            choques,
            None,
            acciones_post_meta,
            detencion_prematura,
            ultima_llegada,
            None,
            None,
            posicion_final,
            direccion_final
        )
        # guardamos las referencias crudas en formato de numpy arrays planos traídos de la GPU
        obj._bloques_giros_raw = bloques_giros_raw
        obj._nb = nb
        obj._llegadas_efectivas_raw = llegadas_efectivas_raw
        obj._nl = nl
        obj._trayectoria_raw = trayectoria_raw
        
        obj._resolved_bloques_giros = None
        obj._resolved_llegadas_efectivas = None
        obj._resolved_trayectoria = None
        return obj

    @property
    def bloques_giros(self) -> Tuple[int, ...]:
        # decodificamos los giros solo en el momento que alguien lo solicite (evaluación perezosa)
        if self._resolved_bloques_giros is None:
            self._resolved_bloques_giros = tuple(self._bloques_giros_raw[:self._nb].tolist())
        return self._resolved_bloques_giros

    @property
    def llegadas_efectivas(self) -> Tuple[int, ...]:
        # decodificamos los pasos de llegada solo en el momento que se solicite
        if self._resolved_llegadas_efectivas is None:
            self._resolved_llegadas_efectivas = tuple(self._llegadas_efectivas_raw[:self._nl].tolist())
        return self._resolved_llegadas_efectivas

    @property
    def trayectoria(self) -> Tuple[Tuple[int, int], ...]:
        # decodificar la trayectoria (lista de coordenadas x,y) es muy costoso en tiempo de cpu en cada generación.
        # la guardamos como un array plano de numpy y solo la convertimos a tuplas de python cuando se consulta
        if self._resolved_trayectoria is None:
            self._resolved_trayectoria = tuple(map(tuple, self._trayectoria_raw.tolist()))
        return self._resolved_trayectoria

    def __getitem__(self, item):
        # mapeamos los índices de namedtuple por si el orquestador accede mediante desempaquetado posicional
        if item == 5:
            return self.bloques_giros
        elif item == 9:
            return self.llegadas_efectivas
        elif item == 10:
            return self.trayectoria
        return super().__getitem__(item)

    def __hash__(self) -> int:
        return hash((
            self.distancia_final,
            self.tau,
            self.es_valido,
            self.pausas_intermedias,
            self.choques,
            self.acciones_post_meta,
            self.detencion_prematura,
            self.ultima_llegada,
            self.posicion_final,
            self.direccion_final
        ))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, MetricasCromosoma):
            return False
        return (
            self.distancia_final == other.distancia_final and
            self.tau == other.tau and
            self.es_valido == other.es_valido and
            self.pausas_intermedias == other.pausas_intermedias and
            self.choques == other.choques and
            self.acciones_post_meta == other.acciones_post_meta and
            self.detencion_prematura == other.detencion_prematura and
            self.ultima_llegada == other.ultima_llegada and
            self.posicion_final == other.posicion_final and
            self.direccion_final == other.direccion_final
        )


def simular_poblacion_acelerada(
    poblacion: List[Cromosoma],
    mapa: np.ndarray,
    inicio: Tuple[int, int],
    meta: Tuple[int, int],
) -> List[MetricasCromosoma]:
    
    # el orquestador principal de la gpu: mueve los datos a la tarjeta, configura la grilla, corre el kernel y descarga las respuestas
    N = len(poblacion)
    n = len(poblacion[0])
    filas, cols = mapa.shape
    inicio_r, inicio_c = inicio
    meta_r, meta_c = meta
    
    # 1. traducción ultra-rápida: en vez de iterar uno a uno en python, concatenamos todos los genes en un string en C,
    # lo leemos a nivel de bytes como array numpy contiguo y mapeamos los caracteres usando el LOOKUP_GENES vectorizado.
    # esto reduce a milisegundos la traducción genotipo-fenotipo
    joined_genes = "".join("".join(c.genes) for c in poblacion)
    arr_bytes = np.frombuffer(joined_genes.encode('ascii'), dtype=np.uint8).reshape(N, n)
    poblacion_num = np.ascontiguousarray(LOOKUP_GENES[arr_bytes], dtype=np.int8)
    
    # 2. caché del mapa en vram: subimos la matriz del mapa a la gpu solo la primera vez y reciclamos el puntero.
    # esto evita transferir el mapa completo por el bus PCIe en cada una de las generaciones
    mapa_num = obtener_mapa_num(mapa)
    mapa_id = id(mapa)
    if mapa_id not in _cache_d_mapa:
        _cache_d_mapa[mapa_id] = cuda.to_device(mapa_num)
    d_mapa = _cache_d_mapa[mapa_id]
    
    # 3. preparamos arrays contiguos en ram con valores centinela (-1) para indicar celdas vacías/no-alcanzadas
    llegadas_efectivas_host = np.ascontiguousarray(np.full((N, n), -1, dtype=np.int32), dtype=np.int32)
    bloques_giros_host = np.ascontiguousarray(np.full((N, n), -1, dtype=np.int32), dtype=np.int32)
    
    # 4. reciclamos los arreglos de salida de la gpu usando el pool pre-asignado para evitar overhead de drivers nvidia
    dev_arrays = obtener_device_arrays(N, n)
    d_poblacion = dev_arrays["d_poblacion"]
    d_llegadas_efectivas = dev_arrays["d_llegadas_efectivas"]
    d_bloques_giros = dev_arrays["d_bloques_giros"]
    
    # copiamos los inputs directamente a la memoria de la gpu
    d_poblacion.copy_to_device(poblacion_num)
    d_llegadas_efectivas.copy_to_device(llegadas_efectivas_host)
    d_bloques_giros.copy_to_device(bloques_giros_host)
    
    # referenciamos los device pointers para las salidas
    d_distancia_final = dev_arrays["d_distancia_final"]
    d_tau = dev_arrays["d_tau"]
    d_es_valido = dev_arrays["d_es_valido"]
    d_pausas_intermedias = dev_arrays["d_pausas_intermedias"]
    d_choques = dev_arrays["d_choques"]
    d_acciones_post_meta = dev_arrays["d_acciones_post_meta"]
    d_detencion_prematura = dev_arrays["d_detencion_prematura"]
    d_ultima_llegada = dev_arrays["d_ultima_llegada"]
    d_p_r_final = dev_arrays["d_p_r_final"]
    d_p_c_final = dev_arrays["d_p_c_final"]
    d_d_final = dev_arrays["d_d_final"]
    d_num_llegadas = dev_arrays["d_num_llegadas"]
    d_num_bloques = dev_arrays["d_num_bloques"]
    d_trayectorias = dev_arrays["d_trayectorias"]
    
    # 5. configuramos la ejecución paralela: 256 hilos por bloque de la gpu
    threadsperblock = 256
    blockspergrid = (N + (threadsperblock - 1)) // threadsperblock
    
    # 6. disparamos el kernel paralelo en la gpu
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
    
    # forzamos a que la cpu se detenga y espere a que todos los núcleos de la gpu terminen de calcular
    cuda.synchronize()
    
    # 7. traemos los resultados brutos de vuelta a la memoria ram
    dist_arr = d_distancia_final.copy_to_host()
    tau_arr = d_tau.copy_to_host()
    valido_arr = d_es_valido.copy_to_host()
    pausas_arr = d_pausas_intermedias.copy_to_host()
    choques_arr = d_choques.copy_to_host()
    post_meta_arr = d_acciones_post_meta.copy_to_host()
    detencion_arr = d_detencion_prematura.copy_to_host()
    ult_llegada_arr = d_ultima_llegada.copy_to_host()
    pr_arr = d_p_r_final.copy_to_host()
    pc_arr = d_p_c_final.copy_to_host()
    df_arr = d_d_final.copy_to_host()
    
    num_llegadas_arr = d_num_llegadas.copy_to_host()
    llegadas_arr = d_llegadas_efectivas.copy_to_host()
    num_bloques_arr = d_num_bloques.copy_to_host()
    bloques_arr = d_bloques_giros.copy_to_host()
    trayectorias_arr = d_trayectorias.copy_to_host()
    
    # 8. armamos los objetos metricascromosoma con los resultados que nos trajo la gpu de forma perezosa (lazy)
    resultados = []
    for i in range(N):
        nl = num_llegadas_arr[i]
        nb = num_bloques_arr[i]
        posicion_final = (int(pr_arr[i]), int(pc_arr[i]))
        direccion_final = DIR_MAP_REV[int(df_arr[i])]
        
        m = LazyMetricasCromosoma(
            distancia_final=int(dist_arr[i]),
            tau=int(tau_arr[i]),
            es_valido=bool(valido_arr[i]),
            pausas_intermedias=int(pausas_arr[i]),
            choques=int(choques_arr[i]),
            bloques_giros_raw=bloques_arr[i],
            nb=nb,
            acciones_post_meta=int(post_meta_arr[i]),
            detencion_prematura=int(detencion_arr[i]),
            ultima_llegada=int(ult_llegada_arr[i]),
            llegadas_efectivas_raw=llegadas_arr[i],
            nl=nl,
            trayectoria_raw=trayectorias_arr[i],
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
    
    # simula un solo cromosoma en la gpu de manera acelerada y transparente
    #
    # magia negra: usamos inspect para robar la variable de la ram (poblacion) de la pila de llamadas.
    # esto nos permite batchar toda la simulación en un solo viaje de gpu sin tocar la orquestacion de cpu
    frame = inspect.currentframe()
    poblacion = None
    try:
        while frame:
            locals_dict = frame.f_locals
            if "poblacion" in locals_dict:
                val = locals_dict["poblacion"]
                if isinstance(val, list) and len(val) > 0 and isinstance(val[0], Cromosoma):
                    poblacion = val
                    break
            frame = frame.f_back
    finally:
        # eliminamos referencias al frame para evitar fugas de memoria en python
        del frame
        
    if poblacion is None:
        # si no se encuentra la lista de la población (ej: pruebas unitarias sueltas), simulamos un solo cromosoma en GPU
        return simular_poblacion_acelerada([cromosoma], mapa, inicio, meta)[0]
        
    pob_id = id(poblacion)
    
    # si es una población nueva (es decir, una nueva generación), limpiamos el caché viejo y mandamos el lote completo a la gpu de una sola vez
    if pob_id not in _poblacion_cache:
        _poblacion_cache.clear()
        metricas_lote = simular_poblacion_acelerada(poblacion, mapa, inicio, meta)
        _poblacion_cache[pob_id] = {id(c): m for c, m in zip(poblacion, metricas_lote)}
        
    crom_id = id(cromosoma)
    # las siguientes n-1 llamadas del ciclo de python leen al instante el resultado desde el caché en ram
    if crom_id in _poblacion_cache[pob_id]:
        return _poblacion_cache[pob_id][crom_id]
        
    # fallback por seguridad por si el cromosoma no estaba en la lista recuperada
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
    
    # corre la evolución completa tanto en cpu secuencial como en gpu cuda de forma silenciosa para medir rendimiento
    import random
    import orquestacion
    from cromosoma import simular
    
    # 1. calentar GPU / compiler
    try:
        c_test = Cromosoma.aleatorio(n, random.Random(seed))
        simular_acelerado(c_test, mapa, inicio, meta)
    except Exception:
        pass
        
    params = {
        "n": n,
        "pm": pm,
        "N": N,
        "G": G,
        "ps": ps,
        "seed": seed
    }
    
    # 2. medir cpu secuencial
    print("corriendo el ciclo evolutivo secuencial en la cpu...")
    t_inicio_sec = time.perf_counter()
    orquestacion.main(simulador_fn=simular, params=params, silencioso=True)
    t_sec = time.perf_counter() - t_inicio_sec
    print(f"completado en cpu: {t_sec:.5f} s")
    
    # 3. medir gpu acelerada
    print("corriendo el ciclo evolutivo acelerado en la gpu...")
    t_inicio_acc = time.perf_counter()
    orquestacion.main(simulador_fn=simular_acelerado, params=params, silencioso=True)
    t_acc = time.perf_counter() - t_inicio_acc
    print(f"completado en gpu: {t_acc:.5f} s")
    
    speedup = t_sec / t_acc if t_acc > 0 else 1.0
    
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
