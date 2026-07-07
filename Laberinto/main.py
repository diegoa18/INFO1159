import os
import sys
import orquestacion

# Auto-detect and set CUDA_HOME to the pip-installed NVIDIA CUDA compiler path if not already configured
if "CUDA_HOME" not in os.environ:
    local_cuda = os.path.expanduser("~/.local/lib/python3.14/site-packages/nvidia/cuda_nvcc")
    if os.path.exists(local_cuda): os.environ["CUDA_HOME"] = local_cuda

if __name__ == "__main__":
    print("Selecciona el modo de ejecución:")
    print("1) ejecutar version secuencial estandar ")
    print("2) Ejecutar version acelerada con GPU NVIDIA CUDA (solo matus)")
    print("3) correr Benchmark de aceleración (solo matus)")

    opcion = input("ingresa tu opción 1, 2 o 3 (Por defecto 1): ").strip()

    if opcion == "2":
        from aceleracion_gpu import simular_acelerado
        orquestacion.main(simulador_fn=simular_acelerado)
    elif opcion == "3":
        import parser_csv
        from aceleracion_gpu import correr_benchmark_completo
        
        ruta_csv, n, pm, N, G, ps, seed = parser_csv.pedir_inputs()
        
        mapa, inicio, meta = parser_csv.cargar_laberinto(ruta_csv)
        
        correr_benchmark_completo(mapa, inicio, meta, n, N, G, pm, ps, seed)
    else:
        orquestacion.main()