import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import norm

EPS = 1e-9  # 0.000000001 -> epsilon numerico para evitar errores de precision


def normalizar(valor):  # redondear a 0 o a 6 decimales
    return 0.0 if abs(valor) < EPS else round(valor, 6)


def es_eje(r):
    a, b, c, _ = r
    return abs(c) < EPS and (
        abs(a) < EPS or abs(b) < EPS
    )  # es decir si a=0 o b=0 si c=0


def parse_num(x):
    # parse para visualizacion sin decimales en las restricciones si es necessario como 3.0 -> 3
    return f"{x:.1f}" if x != int(x) else f"{int(x)}"


def son_paralelas(r1, r2):
    # [:2] para ignorar c y asi sacar determinante con (an,bn), si det = 0 -> ||
    (a1, b1), (a2, b2) = r1[:2], r2[:2]
    return abs(a1 * b2 - a2 * b1) < EPS


def calcular_interseccion(r1, r2):
    if son_paralelas(r1, r2):  # no calcular si ||
        return None

    (a1, b1, c1), (a2, b2, c2) = r1[:3], r2[:3]  # desempaquetar restriccion
    det = a1 * b2 - a2 * b1  # det

    if abs(det) < EPS:
        return None

    return (
        (c1 * b2 - c2 * b1) / det,
        (a1 * c2 - a2 * c1) / det,
    )  # obtener tupla de interseccion


# origen -> a -> b | orientacion de giro y mantener un sentido
def orientacion(origen, a, b):
    return (a[0] - origen[0]) * (b[1] - origen[1]) - (a[1] - origen[1]) * (
        b[0] - origen[0]
    )  # positivo -> izquierda


# convex
def convex_hull(puntos):
    if len(puntos) < 3:
        return list(puntos)

    pts = sorted(puntos)
    lower, upper = [], []

    # almenos 2 anteriores y determinar orientacion
    for p in pts:
        while len(lower) >= 2 and orientacion(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(pts):
        while len(upper) >= 2 and orientacion(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    # ambos lados eliminando lo extremos para evitar duplicados
    return lower[:-1] + upper[:-1]


# diccionario de funciones para interpretar inecuaciones
# a c se le opera con EPS dependiendo de la inecuancion en casos donde v == c pero de false erroneamente por floats
CUMPLE = {
    "<=": lambda v, c: v <= c + EPS,
    ">=": lambda v, c: v >= c - EPS,
}


def cumple_restriccion(punto, restriccion):
    x, y = punto
    a, b, c, signo = restriccion
    # ssi no existe operador -> false, si no cumple la operacion -> falsex
    return CUMPLE.get(signo, lambda v, c: False)(a * x + b * y, c)


def es_factible(punto, restricciones):
    return all(cumple_restriccion(punto, r) for r in restricciones)


def calcular_vertices(restricciones):
    all_v, factibles, seen = [], [], set()
    hull = []
    lineas = [(a, b, c) for a, b, c, _ in restricciones]

    for i, l1 in enumerate(lineas):
        for l2 in lineas[i + 1 :]:
            p = calcular_interseccion(l1, l2)
            if p is None:
                continue

            px, py = map(normalizar, p)
            key = (px, py)

            if key in seen:
                continue

            seen.add(key)
            all_v.append((px, py))

            if es_factible((px, py), restricciones):
                factibles.append((px, py))
                hull = (
                    convex_hull(factibles) if len(factibles) >= 3 else factibles.copy()
                )

    return all_v, factibles, hull


def evaluar_objetivo(vertices, a, b):  # evaluar Z con ciertos vertices
    return [(v, a * v[0] + b * v[1]) for v in vertices]  # retorna tuplas -> ((x,y), Z)


def no_acotado(hull, a, b):
    if len(hull) < 3:
        return True

    d = np.array([a, b], float)
    d /= norm(d) + EPS

    for p1, p2 in zip(hull, hull[1:] + [hull[0]]):
        edge = np.subtract(p2, p1)
        normal = np.array([-edge[1], edge[0]])
        normal /= norm(normal) + EPS

        if np.dot(normal, d) > EPS:
            return False

    return True


def clasificar_hull(hull, a_obj, b_obj):
    n = len(hull)

    if n == 0:
        return "infactible"

    if abs(a_obj) < EPS and abs(b_obj) < EPS:
        raise ValueError("Función objetivo inválida")

    if n < 3:
        return {
            1: "optimo_degenerado_punto",
            2: "optimo_degenerado_segmento",
        }[n]

    area = abs(
        sum(x1 * y2 - x2 * y1 for (x1, y1), (x2, y2) in zip(hull, hull[1:] + [hull[0]]))
    )
    if area < EPS:
        return "degenerado"

    if no_acotado(hull, a_obj, b_obj):
        return "no_acotado"

    return "optimo"


def solucion_base(a_obj, b_obj, tipo, estado):
    return {
        "estado": estado,
        "all_vertices": [],
        "vertices_factibles": [],
        "hull": [],
        "resultados": [],
        "coeficientes": (a_obj, b_obj),
        "tipo": tipo,
        "optimo": None,
        "valor_optimo": None,
    }


# solver
def resolver_PL(restricciones, a_obj, b_obj, tipo="max"):
    if not restricciones:
        return solucion_base(a_obj, b_obj, tipo, "no_acotado")

    all_v, _, hull = calcular_vertices(restricciones)
    resultados = evaluar_objetivo(hull, a_obj, b_obj)
    estado = clasificar_hull(hull, a_obj, b_obj)

    sol = solucion_base(a_obj, b_obj, tipo, estado)

    sol["all_vertices"] = all_v
    sol["vertices_factibles"] = hull
    sol["hull"] = hull
    sol["resultados"] = resultados

    if estado == "optimo" or estado.startswith("optimo_degenerado"):
        opt = (
            max(resultados, key=lambda r: r[1])
            if tipo == "max"
            else min(resultados, key=lambda r: r[1])
        )
        sol["optimo"] = opt
        sol["valor_optimo"] = opt[1]

    return sol


# VISUALIZACION

MARGEN_LINEAS_Z = 1


def fmt_num(numero):
    if numero == 0:
        return "0"
    valor_absoluto = abs(numero)
    if valor_absoluto >= 1000 or valor_absoluto <= EPS:
        return f"{numero:.2e}"
    elif abs(numero - int(numero)) < EPS:
        return f"{int(numero)}"
    else:
        return f"{numero:.2f}"


def fmt_objetivo(coeficiente_x, coeficiente_y):
    terminos = []
    for coeficiente, variable in ((coeficiente_x, "x"), (coeficiente_y, "y")):
        if abs(coeficiente) < EPS:
            continue
        termino = fmt_num(abs(coeficiente)) + variable
        terminos.append(
            (termino if coeficiente > 0 else f"-{termino}")
            if not terminos
            else (f"+ {termino}" if coeficiente > 0 else f"- {termino}")
        )
    return " ".join(terminos) if terminos else "0"


def construir_titulo(solucion):
    (coordenada_x, coordenada_y), valor_optimo = solucion["optimo"]
    coeficiente_x, coeficiente_y = solucion["coeficientes"]
    expresion_objetivo = fmt_objetivo(coeficiente_x, coeficiente_y)
    return f"Z = {expresion_objetivo}  |  en ({fmt_num(coordenada_x)}, {fmt_num(coordenada_y)}) → Z = {fmt_num(valor_optimo)}"


def calcular_limites(vertices, margen_pct=0.1):
    coordenadas_x, coordenadas_y = zip(*vertices)
    xmin, xmax = min(coordenadas_x), max(coordenadas_x)
    ymin, ymax = min(coordenadas_y), max(coordenadas_y)
    dx = (xmax - xmin) or max(abs(xmax), 1.0)
    dy = (ymax - ymin) or max(abs(ymax), 1.0)
    return (
        xmin - dx * margen_pct,
        xmax + dx * margen_pct,
        ymin - dy * margen_pct,
        ymax + dy * margen_pct,
    )


def puntos_recta(restriccion, xmin, xmax, ymin, ymax, num_points: int = 200):
    coeficiente_x, coeficiente_y, termino_independiente, _ = restriccion

    resolucion = max((xmax - xmin), (ymax - ymin)) / num_points

    if abs(coeficiente_y) < EPS:
        return [
            (termino_independiente / coeficiente_x, coordenada_y)
            for coordenada_y in np.linspace(ymin, ymax, 3)
        ]

    coordenadas_x = np.linspace(xmin, xmax, int((xmax - xmin) / resolucion) + 1)
    coordenadas_y = (
        termino_independiente - coeficiente_x * coordenadas_x
    ) / coeficiente_y  # y

    return list(
        zip(
            coordenadas_x[np.isfinite(coordenadas_y)],
            coordenadas_y[np.isfinite(coordenadas_y)],
        )
    )  # pares (eliminando valores invalidos)


def formatear_restriccion(restriccion):
    coeficiente_x, coeficiente_y, termino_independiente, signo = restriccion
    return f"{parse_num(coeficiente_x)}x {'+' if coeficiente_y >= 0 else '-'}{parse_num(abs(coeficiente_y))}y {signo} {parse_num(termino_independiente)}"


def configurar_ejes(eje, limites):
    eje.set_xlim(limites[:2])
    eje.set_ylim(limites[2:])
    eje.set_aspect("auto")
    eje.ticklabel_format(style="sci", axis="both", scilimits=(0, 0))
    eje.grid(True, alpha=0.3, linestyle="--")
    eje.set_xlabel("x", fontsize=11)
    eje.set_ylabel("y", fontsize=11)


def graficar_restricciones(eje, restricciones, limites):
    mapa_colores = plt.colormaps["tab10"]
    for indice, restriccion in enumerate(restricciones):
        puntos = puntos_recta(restriccion, *limites)
        if not puntos:
            continue
        coordenadas_x, coordenadas_y = zip(*puntos)
        if es_eje(restriccion):
            eje.plot(
                coordenadas_x, coordenadas_y, color="black", linewidth=2.5, alpha=0.9
            )
        else:
            eje.plot(
                coordenadas_x,
                coordenadas_y,
                color=mapa_colores(indice % 10),
                linestyle="--",
                linewidth=1.5,
                alpha=0.8,
                label=formatear_restriccion(restriccion),
            )


def graficar_region(eje, hull):
    if len(hull) < 3:
        return
    coordenadas_x, coordenadas_y = zip(*(hull + [hull[0]]))
    eje.fill(coordenadas_x, coordenadas_y, color="lightgreen", alpha=0.3, zorder=1)
    eje.plot(
        coordenadas_x,
        coordenadas_y,
        color="green",
        linewidth=2.5,
        alpha=0.9,
        zorder=2,
        label="Región Factible",
    )


def graficar_vertices(eje, vertices_totales, vertices_factibles, optimo):
    conjunto_factibles = set(vertices_factibles)
    vertices_no_factibles = [
        vertice for vertice in vertices_totales if vertice not in conjunto_factibles
    ]
    if vertices_no_factibles:
        coordenadas_x, coordenadas_y = zip(*vertices_no_factibles)
        eje.scatter(
            coordenadas_x,
            coordenadas_y,
            c="gray",
            s=50,
            alpha=0.8,
            zorder=3,
            label="Vértices no factibles",
        )
    for vertice in vertices_factibles:
        if optimo is not None and vertice == optimo[0]:
            eje.scatter(
                *vertice,
                c="gold",
                s=300,
                zorder=8,
                edgecolors="orange",
                linewidths=2,
                marker="*",
                label="Óptimo",
            )
        else:
            eje.scatter(*vertice, c="red", s=80, zorder=5, edgecolors="darkred")


def graficar_Z(eje, resultados, coeficiente_x, coeficiente_y, limites, optimo):
    xmin, xmax, ymin, ymax = limites
    valores_x = np.linspace(xmin, xmax, 200)
    for vertice, valor_z in resultados:
        es_optimo = vertice == optimo[0]
        color = "blue" if es_optimo else "steelblue"
        estilo_linea = "-" if es_optimo else "--"
        ancho_linea = 2.5 if es_optimo else 1.2
        transparencia = 1.0 if es_optimo else 0.4
        profundidad = 6 if es_optimo else 4
        if abs(coeficiente_y) > EPS:
            valores_y = (valor_z - coeficiente_x * valores_x) / coeficiente_y
            mascara = (valores_y >= ymin - MARGEN_LINEAS_Z) & (
                valores_y <= ymax + MARGEN_LINEAS_Z
            )
            if mascara.any():
                eje.plot(
                    valores_x[mascara],
                    valores_y[mascara],
                    color=color,
                    linewidth=ancho_linea,
                    alpha=transparencia,
                    linestyle=estilo_linea,
                    zorder=profundidad,
                )
        else:
            eje.plot(
                [valor_z / coeficiente_x] * 2,
                [ymin, ymax],
                color=color,
                linewidth=ancho_linea,
                alpha=transparencia,
                linestyle=estilo_linea,
                zorder=profundidad,
            )


def graficar_etiquetas(eje, resultados, optimo):
    for vertice, valor_z in resultados:
        es_optimo = vertice == optimo[0]
        eje.annotate(
            f"Z = {fmt_num(valor_z)}",
            vertice,
            textcoords="offset points",
            xytext=(8, 8),
            fontsize=(9 if es_optimo else 8),
            color=("darkgreen" if es_optimo else "dimgray"),
            fontweight=("bold" if es_optimo else "normal"),
            bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7),
            zorder=7,
        )


def graficar(solucion, restricciones):
    estado = solucion["estado"]

    if estado != "optimo":
        return

    hull = solucion["hull"]
    limites = calcular_limites(hull)
    xmin, xmax, ymin, ymax = limites

    rango = max(xmax - xmin, ymax - ymin, 1)
    escala = max(1, int(rango))
    figsize = (10 + escala, 8 + escala)
    figsize = tuple(min(x, 20) for x in figsize)

    figura, eje = plt.subplots(figsize=figsize)
    configurar_ejes(eje, limites)
    graficar_restricciones(eje, restricciones, limites)
    graficar_region(eje, hull)
    optimo = solucion["optimo"]
    graficar_vertices(
        eje, solucion["all_vertices"], solucion["vertices_factibles"], optimo
    )
    graficar_Z(eje, solucion["resultados"], *solucion["coeficientes"], limites, optimo)
    graficar_etiquetas(eje, solucion["resultados"], optimo)
    titulo = construir_titulo(solucion)

    eje.set_title(titulo, fontsize=14, fontweight="bold", pad=12)
    eje.legend(loc="best", fontsize=9, framealpha=0.9)
    plt.tight_layout()
    plt.show()
