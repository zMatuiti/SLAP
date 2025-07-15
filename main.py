import pandas as pd
import numpy as np
import random
import pyvista as pv
from matplotlib import cm
import matplotlib.colors as mcolors

class Almacen:
    def __init__(self, dataset_pth, affinity_pth, num_racks, lenght_rack, height_rack, num_productos_populares):
        print("Iniciando la creacion del modelo de almacen...")
        
        # estructura fisica del almacen
        self.ancho = lenght_rack      # eje X
        self.profundidad = num_racks * 2 - 1  # eje Y
        self.alto = height_rack       # eje Z
        self.x_corredor = 0           # Corredor principal en x=0 (o self.ancho - 1)
        self.punto_io = (self.x_corredor, self.profundidad // 2, 0)  # Punto medio del corredor
        
        # Diccionario para guardar asignaciones: {'producto_id': (x, y, z)}
        self.ubicaciones = {}
        
        self.dataset, self.affinity = self._load_dataset_and_affinity(dataset_pth, affinity_pth, num_productos_populares)
        print(f"\nSe analizaron los {num_productos_populares} productos más populares.")

        self.ordenes_simulacion = self._crear_ordenes_de_simulacion()
        print(f"Se crearon {len(self.ordenes_simulacion)} órdenes de simulación para las pruebas.")

    def _load_dataset_and_affinity(self, dataset_fp, affinity_fp, num_productos):
        print(f"\nCargando datos desde: {dataset_fp}...")
        dataset = pd.read_csv(dataset_fp)

        # Calcular la frecuencia de cada producto
        frecuencia = dataset['StockCode'].value_counts()
        productos_populares = frecuencia.head(num_productos).index

        # Filtrar dataset con productos populares
        dataset_filtrado = dataset[dataset['StockCode'].isin(productos_populares)]
        self.datos_demanda = frecuencia.loc[productos_populares]
        print(f"Datos de demanda:\n{self.datos_demanda}")

        # Cargar y filtrar matriz de afinidad
        affinity = pd.read_csv(affinity_fp, index_col=0)
        affinity.index = affinity.index.astype(str)
        affinity.columns = affinity.columns.astype(str)
        productos_populares = productos_populares.astype(str)
        affinity_filtrada = affinity.loc[productos_populares, productos_populares]

        print("Productos populares:", list(productos_populares))
        print(f"Matriz de afinidad filtrada con {len(affinity_filtrada)} productos.")
        print(f"Matriz de afinidad:\n{affinity_filtrada}")

        return dataset_filtrado, affinity_filtrada
        
    def _crear_ordenes_de_simulacion(self):
        ordenes = []
        productos_populares = self.datos_demanda.index.tolist()

        for _ in range(100):
            tamano_orden = np.random.randint(2, 6)
            orden = np.random.choice(productos_populares, size=tamano_orden, replace=False).tolist()
            ordenes.append(orden)

        return ordenes

    def _calcular_distancia_manhattan(self, punto1, punto2):
        return sum(abs(p1 - p2) for p1, p2 in zip(punto1, punto2))
    
    def _calcular_distancia_realista(self, punto1, punto2):
        """
        Calcula la distancia que recorrería un trabajador usando pasillos.
        Se asume:
        - Movimiento vertical (z) se hace en el lugar.
        - Para cambiar de fila (y), hay que volver al corredor (eje x fijo, por ejemplo x=0 o x=max).
        """
        x1, y1, z1 = punto1
        x2, y2, z2 = punto2

        if y1 == y2:
            # Están en el mismo pasillo
            distancia = abs(x1 - x2) + abs(z1 - z2)
        else:
            # Distancia hasta corredor (x fijo, ej. x=0), ida y vuelta
            corredor_x = 0  # o self.ancho - 1
            distancia = abs(x1 - corredor_x)  # salir del pasillo
            distancia += abs(y1 - y2)         # caminar por corredor
            distancia += abs(x2 - corredor_x) # entrar al nuevo pasillo
            distancia += abs(z1 - z2)         # ajustar altura

        return distancia

    def asignar_producto(self, producto_id, coordenadas):
        x, y, z = coordenadas
        if (0 <= x < self.ancho) and (0 <= y < self.profundidad) and (0 <= z < self.alto):
            self.ubicaciones[producto_id] = coordenadas
        else:
            print(f"Error: Coordenadas {coordenadas} fuera de los límites del almacén.")

    def calcular_costo_total_simulacion(self):
        distancia_total = 0
        if not self.ubicaciones:
            return float('inf')

        for orden in self.ordenes_simulacion:
            punto_actual = self.punto_io
            distancia_orden = 0

            for producto_id in orden:
                if producto_id in self.ubicaciones:
                    destino = self.ubicaciones[producto_id]
                    distancia_orden += self._calcular_distancia_realista(punto_actual, destino)
                    punto_actual = destino

            distancia_orden += self._calcular_distancia_realista(punto_actual, self.punto_io)
            distancia_total += distancia_orden

        return distancia_total / len(self.ordenes_simulacion)
    
    def visualizar_almacen(self):
        plotter = pv.Plotter()

        # Dimensiones de cada cubo (rack)
        dx, dy, dz = 1.0, 1.0, 1.0

        # Generate a colormap for the products
        productos = list(self.ubicaciones.keys())
        colormap = cm.get_cmap('tab20', len(productos))  # Use a colormap with enough colors
        norm = mcolors.Normalize(vmin=0, vmax=len(productos) - 1)

        for x in range(self.ancho):
            for y in range(0, self.profundidad, 2):
                for z in range(self.alto):
                    # Crear un cubo para cada posición del rack
                    cubo = pv.Cube(center=(x + dx/2, y + dy/2, z + dz/2), x_length=dx, y_length=dy, z_length=dz)
                    plotter.add_mesh(cubo, color='lightgray', opacity=0.1, show_edges=True)

        # Agregar cubos por cada producto (racks)
        for idx, (producto_id, (x, y, z)) in enumerate(self.ubicaciones.items()):
            color = mcolors.to_hex(colormap(norm(idx)))  # Convert colormap value to hex color
            cubo = pv.Cube(center=(x + dx/2, y + dy/2, z + dz/2), x_length=dx, y_length=dy, z_length=dz)
            plotter.add_mesh(cubo, color=color, opacity=1.0, show_edges=True)

        for y in range(self.profundidad):
            base = pv.Cube(center=(-1 + dx/2, y + dy/2, 0), x_length=dx, y_length=dy, z_length=0.05)
            plotter.add_mesh(base, color='yellow', opacity=1.0)
            if y % 2 == 1:
                for x in range(self.ancho):
                    corredor = pv.Cube(center=(x + dx/2, y + dy/2, 0), x_length=dx, y_length=dy, z_length=0.05)
                    plotter.add_mesh(corredor, color='yellow', opacity=1)


        # Agregar punto de entrada/salida
        x0, y0, z0 = self.punto_io
        entrada = pv.Sphere(radius=0.2, center=(x0 + dx/2 - 1, y0 + dy/2, z0 + dz/2))
        plotter.add_mesh(entrada, color='red')

        plotter.show()
    
class GeneticOptimizer:
    def __init__(self, almacen, poblacion=50, generaciones=100, prob_mutacion=0.1):
        self.almacen = almacen
        self.productos = almacen.datos_demanda.index.tolist()
        self.poblacion_size = poblacion
        self.generaciones = generaciones
        self.prob_mutacion = prob_mutacion
        self.posibles_ubicaciones = self._generar_posibles_ubicaciones()

        if len(self.posibles_ubicaciones) < len(self.productos):
            raise ValueError(
                f"No hay suficientes ubicaciones disponibles ({len(self.posibles_ubicaciones)}) "
                f"para asignar todos los productos ({len(self.productos)})."
            )
    
    def _generar_posibles_ubicaciones(self):
        ubicaciones = []
        for x in range(self.almacen.ancho):
            for y in range(0, self.almacen.profundidad, 2):
                for z in range(self.almacen.alto):
                    ubicaciones.append((x, y, z))
        return ubicaciones

    def _crear_individuo(self):
        ubicaciones = random.sample(self.posibles_ubicaciones, len(self.productos))
        return dict(zip(self.productos, ubicaciones))

    def _fitness(self, individuo, alpha=0.7, beta=0.3):
        self.almacen.ubicaciones = individuo

        # 1. Costo de picking
        costo_picking = self.almacen.calcular_costo_total_simulacion()

        # 2. Costo de afinidad
        costo_afinidad = 0
        matriz = self.almacen.affinity

        for i in range(len(self.productos)):
            for j in range(i + 1, len(self.productos)):
                p1, p2 = self.productos[i], self.productos[j]
                afinidad = matriz.loc[p1, p2] if p1 in matriz.columns and p2 in matriz.columns else 0

                if afinidad > 0:
                    loc1 = individuo[p1]
                    loc2 = individuo[p2]
                    distancia = self.almacen._calcular_distancia_realista(loc1, loc2)
                    costo_afinidad += afinidad * distancia

        # Normalización simple por cantidad de pares
        if len(productos) > 1:
            costo_afinidad /= (len(productos) * (len(productos) - 1) / 2)

        return alpha * costo_picking + beta * costo_afinidad

    def _seleccionar_padres(self, poblacion, scores):  # Selección por torneo
        padres = []
        for _ in range(2):
            torneo = random.sample(list(zip(poblacion, scores)), k=5)
            torneo.sort(key=lambda x: x[1])  # Menor distancia = mejor
            padres.append(torneo[0][0])
        return padres

    def _crossover(self, padre1, padre2):  # Cruce de un punto
        punto = random.randint(1, len(self.productos) - 2)
        productos = self.productos
        hijo = {}

        for i in range(len(productos)):
            producto = productos[i]
            hijo[producto] = padre1[producto] if i < punto else padre2[producto]

        usadas = list(hijo.values())
        ubicaciones_unicas = set()
        duplicados = []

        for producto, loc in hijo.items():
            if loc in ubicaciones_unicas:
                duplicados.append(producto)
            else:
                ubicaciones_unicas.add(loc)

        faltantes = [loc for loc in self.posibles_ubicaciones if loc not in usadas]

        for producto in duplicados:
            if faltantes:
                hijo[producto] = faltantes.pop()
            else:
                # Asignación forzada para evitar crash
                hijo[producto] = random.choice(self.posibles_ubicaciones)

        return hijo

    def _mutar(self, individuo):
        if random.random() < self.prob_mutacion:
            p1, p2 = random.sample(self.productos, 2)
            individuo[p1], individuo[p2] = individuo[p2], individuo[p1]

    def optimizar(self):
        poblacion = [self._crear_individuo() for _ in range(self.poblacion_size)]

        mejor_individuo = None
        mejor_score = float('inf')

        for gen in range(self.generaciones):
            scores = [self._fitness(ind) for ind in poblacion]
            generacion_mejor = min(scores)
            print(f"Generación {gen+1}: Mejor Costo = {generacion_mejor:.2f}")

            # Guardar mejor solución
            idx = scores.index(generacion_mejor)
            if generacion_mejor < mejor_score:
                mejor_score = generacion_mejor
                mejor_individuo = poblacion[idx]

            # Nueva población
            nueva_poblacion = []
            while len(nueva_poblacion) < self.poblacion_size:
                padre1, padre2 = self._seleccionar_padres(poblacion, scores)
                hijo = self._crossover(padre1, padre2)
                self._mutar(hijo)
                nueva_poblacion.append(hijo)

            poblacion = nueva_poblacion

        print(f"\nMejor Costo Encontrado: {mejor_score:.2f}")
        return mejor_individuo

if __name__ == "__main__":
    data_file = 'OnlineRetail_Cleaned.csv'
    affinity_matrix_file = 'affinity_matrix.csv'

    mi_almacen = Almacen(data_file, affinity_matrix_file,
                         num_racks=10,
                         lenght_rack=5,
                         height_rack=3,
                         num_productos_populares=10)

    print("\n--- Demostración de asignación aleatoria ---")
    productos = mi_almacen.datos_demanda.index.tolist()

    optimizador = GeneticOptimizer(mi_almacen, poblacion=50, generaciones=30, prob_mutacion=0.2)
    mejor_asignacion = optimizador.optimizar()

    mi_almacen.ubicaciones = mejor_asignacion

    costo_final = mi_almacen.calcular_costo_total_simulacion()
    print(f"\nCosto final tras optimización: {costo_final:.2f}")

    mi_almacen.visualizar_almacen()