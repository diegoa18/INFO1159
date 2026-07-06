# Diagnóstico de conexión entre módulos

## Grafo de dependencias

```
parser_csv.py (Matus)
  ↓ mapa, inicio, meta  (numpy array, tuplas (fila, col))

cromosoma.py (Diego)
  ↓ MetricasCromosoma, simular()

fitness.py (Dani)
  ↓ funcion_objetivo_J, fitness()

seleccion.py (Joaquín)
  ↓ ordenar_poblacion, seleccionar_padres

orquestacion.py (Débora)
  ↓ main() — coordina todo
```

---

## Problemas por módulo

### 1. Matus (parser_csv) — **un bug crítico**

**Problema:** El módulo ejecuta `input()` al ser importado (líneas 67–86):

```python
print("configuracion del algoritmo genetico")
ruta_csv = input("ingresa la ruta del archivo csv...")
n = int(input("longitud del cromosoma (n): "))
# ...
```

Eso corre cuando cualquier otro módulo hace `import parser_csv`, lo que va a trabar toda la ejecución.

**Solución:**
- Mover todo el bloque de input (líneas 67–86) dentro de `if __name__ == "__main__":`
- La función `cargar_laberinto(ruta_archivo)` está bien tal cual
- `cargar_laberinto` retorna `(mapa_numpy, inicio, meta)` con coordenadas `(fila, col)` — compatible con `simular()`

---

### 2. Dani (fitness) — **debe adaptarse a MetricasCromosoma**

**Problema:** Define su propio `ResultadoEjecucion` en vez de usar `MetricasCromosoma` que exporta `cromosoma.py`.

| Actual (ResultadoEjecucion) | Debe cambiar a (MetricasCromosoma) |
|---|---|
| `resultado.distancia` | `resultado.distancia_final` |
| `resultado.q_prematuros` | `resultado.detencion_prematura` |
| `resultado.llego_meta` → eliminar | No existe en MetricasCromosoma (redundante con `es_valido`) |
| `distancia: float` | `int` (el PDF usa enteros) |
| `tau: float` | `int` |
| `bloques_giros: List[int]` | `Tuple[int, ...]` (o aceptar Sequence) |

**Import que debe usar:**
```python
from cromosoma import MetricasCromosoma
```

**Firma de `funcion_objetivo_J`:**
```python
def funcion_objetivo_J(m: MetricasCromosoma) -> float:
```

Las funciones de penalización individuales (`penalizacion_pausas`, `penalizacion_choques`, etc.) están bien porque reciben enteros sueltos.

---

### 3. Joaquín (selección) — **debe adaptar tipos**

**Problema:** Importa `ResultadoEjecucion` desde fitness (que no existe más) y usa `resultado.llego_meta`.

**Import correcto:**
```python
from cromosoma import Cromosoma, MetricasCromosoma
from fitness import funcion_objetivo_J

Individuo = Tuple[Cromosoma, MetricasCromosoma]
```

**`prioridad_factibilidad`** — `llego_meta` no existe en MetricasCromosoma. Debe ser:

```python
def prioridad_factibilidad(individuo: Individuo) -> int:
    _, m = individuo
    if m.es_valido:
        return 0
    if len(m.llegadas_efectivas) > 0:  # llegó al menos una vez
        return 1
    return 2
```

**`clave_ordenamiento`** — cambiar `resultado.distancia` por `resultado.distancia_final`:

```python
def clave_ordenamiento(individuo: Individuo) -> Tuple[int, float, int, int]:
    _, m = individuo
    return (prioridad_factibilidad(individuo),
            funcion_objetivo_J(m),
            m.distancia_final,
            m.tau)
```

**`elitismo`** — La función actual pone el mejor al final de la lista. El PDF dice:

> Pt+1 = {x\*t} ∪ Ot, donde |Ot| = N−1

El mejor global ocupa **un cupo** en la nueva generación, no se "inserta al final". Sugerencia: que `elitismo` solo retorne el mejor individuo, y que la orquestación (Débora) construya Pt+1 con ese mejor + N−1 descendientes.

---

### 4. Débora (orquestación) — **reemplazar todos los mocks**

Actualmente todo es simulado. Necesita conectar con los módulos reales:

| Mock actual | Debe usar |
|---|---|
| `mock_cargar_parametros()` | `parser_csv.cargar_laberinto(ruta)` |
| `["M","H","M"]` como cromosoma | `Cromosoma.aleatorio(n, rng)` |
| `mock_ejecutar_y_evaluar()` que inventa valores | `simular(c, mapa, inicio, meta)` → `MetricasCromosoma` + `funcion_objetivo_J(m)` |
| `mock_seleccion_elitismo_reproduccion()` | `seleccionar_padres()`, `cruzar_un_punto()`, `mutar()` reales |
| `mejor_global["J"]` (dict) | `m.distancia_final`, `m.tau`, `m.trayectoria` |
| Lista plana como población | `List[Tuple[Cromosoma, MetricasCromosoma]]` |

**Estructura del main:**

```python
def main():
    # 1. Parámetros y mapa
    params, mapa, inicio, meta = parser_csv.cargar_laberinto(ruta_csv)
    rng = random.Random(params["seed"])

    # 2. Población inicial
    poblacion = [Cromosoma.aleatorio(params["n"], rng)
                 for _ in range(params["N"])]

    mejor_global = None
    historico_j = []
    historico_validas = []

    for gen in range(params["G"]):
        # Evaluar
        evaluados = [(c, simular(c, mapa, inicio, meta)) for c in poblacion]

        # Fitness vía Dani
        for c, m in evaluados:
            j = funcion_objetivo_J(m)

        # Ranking vía Joaquín
        ordenados = ordenar_poblacion(evaluados)

        # Mejor global (elitismo)
        mejor_actual = ordenados[0]
        if mejor_global is None or clave_orden(mejor_actual) < clave_orden(mejor_global):
            mejor_global = mejor_actual

        historico_j.append(funcion_objetivo_J(mejor_global[1]))
        historico_validas.append(sum(1 for _, m in evaluados if m.es_valido) / params["N"])

        # Generar nueva población: [mejor_global] + N-1 descendientes
        nueva_pob = [mejor_global]
        while len(nueva_pob) < params["N"]:
            p1, p2 = seleccionar_padres(ordenados, params["ps"])
            c = rng.randint(1, params["n"] - 1)
            h1, h2 = p1[0].cruzar_un_punto(p2[0], c)
            h1.mutar(params["pm"], rng)
            h2.mutar(params["pm"], rng)
            nueva_pob.append((h1, None))  # se evaluará en la próxima gen
            if len(nueva_pob) < params["N"]:
                nueva_pob.append((h2, None))

        poblacion = [c for c, _ in nueva_pob]

    # 3. Reportes
    mostrar_mejores_unicos(mejor_global)
    graficar_evolucion_j(historico_j)
    graficar_proporcion_validas(historico_validas)
```

**Tracking de cromosomas únicos** (para el reporte de "mejores cromosomas únicos"):

```python
mejores_unicos = set()
# ... durante la ejecución ...
if j == mejor_j_global:
    mejores_unicos.add(c)  # funciona gracias a __eq__/__hash__
```

---

## Orden recomendado para integrar

```
1. Matus: arreglar input() al importar parser_csv
       ↓
2. Dani: adaptar fitness.py a MetricasCromosoma
       ↓
3. Joaquín: adaptar seleccion.py a MetricasCromosoma
       ↓
4. Débora: conectar orquestación con los 3 módulos reales
```

El módulo de **Diego (cromosoma)** ya está completo y no necesita cambios. Es la base de la que todos dependen.
