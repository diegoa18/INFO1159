import os
import sys
import orquestacion

# auto-detectamos y seteamos cuda_home para que numba encuentre nvcc instalado en el home del usuario sin tener que configurar variables de entorno a mano en la máquina
if "CUDA_HOME" not in os.environ:
    local_cuda = os.path.expanduser("~/.local/lib/python3.14/site-packages/nvidia/cuda_nvcc")
    if os.path.exists(local_cuda): os.environ["CUDA_HOME"] = local_cuda

if __name__ == "__main__":
    # un menú simple para elegir cómo correr el algoritmo. la opción 1 es secuencial pura en cpu, la 2 es en gpu batchada y la 3 es el test de velocidad
    print("Selecciona el modo de ejecución:")
    print("1) ejecutar version secuencial estandar ")
    print("2) Ejecutar version acelerada con GPU NVIDIA CUDA (solo matus)")
    print("3) correr Benchmark de aceleración (solo matus)")

    opcion = input("ingresa tu opción 1, 2 o 3 (Por defecto 1): ").strip()

    if opcion == "2":
        # si es la opción 2, importamos el simulador gpu y se lo pasamos al main de orquestacion para que reemplace al simulador cpu por defecto sin alterar el ciclo evolutivo
        from aceleracion_gpu import simular_acelerado
        orquestacion.main(simulador_fn=simular_acelerado)
    elif opcion == "3":
        # para el benchmark, pedimos los inputs una sola vez, cargamos el mapa y llamamos al ejecutor que mide los tiempos de cpu y gpu de forma limpia y silenciosa
        import parser_csv
        from aceleracion_gpu import correr_benchmark_completo
        
        ruta_csv, n, pm, N, G, ps, seed = parser_csv.pedir_inputs()
        mapa, inicio, meta = parser_csv.cargar_laberinto(ruta_csv)
        correr_benchmark_completo(mapa, inicio, meta, n, N, G, pm, ps, seed)
    else:
        # por defecto (opción 1), corremos la orquestacion estándar que usa el simulador de cpu de toda la vida
        orquestacion.main()