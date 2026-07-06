# Orquestación algoritmo Genético para la Resolución de Laberintos 
import random
import matplotlib.pyplot as plt


# ---- MOCKS / SIMULACIONES ----

def mock_cargar_parametros():
    # simula lo que hace parser_csv pero sin pedir inputs por consola para no trabar las pruebas
    mapa = [["X","X"], ["X","X"]] # matriz falsa
    return {
        "n": 20, "pm": 0.15, "N": 5, "G": 50, "ps": 0.3, "seed": 42
    }, mapa, (1,1), (2,2)

def mock_ejecutar_y_evaluar(poblacion):
    # simula la ejecucion en el laberinto y el calculo de fitness de la dani
    evaluados = []
    for cromo in poblacion:
        # inventamos valores para probar el ordenamiento 
        j_inventado = random.uniform(10, 500)
        es_valido = random.choice([True, False])
        distancia = random.randint(0, 10)
        tau = random.randint(5, 20)
        
        # guardamos (cromosoma, validez, J, D, tau, ruta_falsa)
     
        ruta_falsa = [(1,1), (1,2), (2,2)]
        evaluados.append({
            "cromo": cromo, 
            "valido": es_valido, 
            "J": j_inventado, 
            "D": distancia, 
            "tau": tau,
            "ruta": ruta_falsa
        })
    return evaluados

def mock_seleccion_elitismo_reproduccion(evaluados, params):
    # retorna una nueva poblacion al azar
    return [["M", "H", "M"] for _ in range(params["N"])]

# ---- ORQUESTACION Y RESULTADOS ----

def graficar_evolucion_j(historico_j):
    # grafica el mejor J por generacion en escala logaritmica (requisito pauta)
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(historico_j) + 1), historico_j, marker='o', linestyle='-', color='b')
    plt.yscale('log')
    plt.title('Evolución del Mejor Valor de Función Objetivo (Global)')
    plt.xlabel('Generación')
    plt.ylabel('Función Objetivo J (Escala Log)')
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.show()

def graficar_proporcion_validas(historico_validas):
    # grafica el porcentaje de soluciones validas por generacion (requisito pauta)
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(historico_validas) + 1), historico_validas, color='g')
    plt.title('Proporción de Soluciones Válidas por Generación')
    plt.xlabel('Generación')
    plt.ylabel('Proporción (0.0 a 1.0)')
    plt.ylim(-0.1, 1.1)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()

def mostrar_mejores_unicos(mejor_global):
    # imprime en consola la info del mejor cromosoma encontrado
    # cuando esten los reales, se filtra por cromosomas unicos que empaten en J
    print("\n" + "="*50)
    print("REPORTE DE MEJORES CROMOSOMAS ÚNICOS")
    print("="*50)
    print(f"Mejor J global encontrado: {mejor_global['J']:.2f}")
    print(f"Pasos tomados (tau): {mejor_global['tau']}")
    print(f"Trayectoria auditada (X, Y):")
    for paso in mejor_global['ruta']:
        print(f" -> {paso}")
    print("="*50)

def main():
    # 1. carga de datos (usando el mock por ahora)
    params, mapa, inicio, meta = mock_cargar_parametros()
    random.seed(params["seed"])
    
    # 2. inicializar poblacion
    poblacion = [["M", "H", "M"] for _ in range(params["N"])]
    
    # variables para guardar estadisticas de las graficas
    historico_mejor_j = []
    historico_prop_validas = []
    
    mejor_global_absoluto = None

    print("Iniciando evolución...")
    
    # 3. ciclo de generaciones
    for gen in range(params["G"]):
        # evaluar poblacion actual
        evaluados = mock_ejecutar_y_evaluar(poblacion)
        
        # buscar el mejor de esta generacion para las estadisticas
        # el orden lexicografico real lo hara Joaco: prioridad valido, menor J, menor D, menor tau
        evaluados.sort(key=lambda x: (not x["valido"], x["J"], x["D"], x["tau"]))
        mejor_gen = evaluados[0]
        
        # actualizar el mejor global historico (Elitismo puro para el reporte)
        if mejor_global_absoluto is None or mejor_gen["J"] < mejor_global_absoluto["J"]:
            mejor_global_absoluto = mejor_gen
            
        historico_mejor_j.append(mejor_global_absoluto["J"])
        
        # calcular proporcion de validas
        cant_validas = sum(1 for ind in evaluados if ind["valido"])
        historico_prop_validas.append(cant_validas / params["N"])
        
        # delegar seleccion, cruza y mutacion para la siguiente generacion
        poblacion = mock_seleccion_elitismo_reproduccion(evaluados, params)

    print("Evolución terminada. Generando reportes...")
    
    # 4. mostrar resultados exigidos por la rubrica
    mostrar_mejores_unicos(mejor_global_absoluto)
    graficar_evolucion_j(historico_mejor_j)
    graficar_proporcion_validas(historico_prop_validas)

if __name__ == "__main__":
    main()