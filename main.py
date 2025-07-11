import pandas as pd
import numpy as np
import random

class Almacen:
    def __init__(self, dataset_pth, affinity_pth, num_racks, lenght_rack, height_rack, num_productos_populares):
        print("Iniciando la creacion del modelo de almacen...")
        
        # estructura fisica del almacen
        self.ancho = lenght_rack      # eje X
        self.profundidad = num_racks  # eje Y
        self.alto = height_rack       # eje Z
        self.total_spaces = self.ancho * self.profundidad * self.alto
        self.punto_io = (0, 0, 0)     # Punto de entrada/salida para el picking
        
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
                    distancia_orden += self._calcular_distancia_manhattan(punto_actual, destino)
                    punto_actual = destino

            distancia_orden += self._calcular_distancia_manhattan(punto_actual, self.punto_io)
            distancia_total += distancia_orden

        return distancia_total / len(self.ordenes_simulacion)
    
class GeneticOptimizer:
    def __init__(self, almacen, poblacion=50, generaciones=100, prob_mutacion=0.1):
        self.almacen = almacen
        self.productos = almacen.datos_demanda.index.tolist()
        self.poblacion_size = poblacion
        self.generaciones = generaciones
        self.prob_mutacion = prob_mutacion
        self.posibles_ubicaciones = self._generar_posibles_ubicaciones()
    
    def _generar_posibles_ubicaciones(self):
        ubicaciones = []
        for x in range(self.almacen.ancho):
            for y in range(self.almacen.profundidad):
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
        productos = self.productos
        matriz = self.almacen.affinity

        for i in range(len(productos)):
            for j in range(i + 1, len(productos)):
                p1, p2 = productos[i], productos[j]
                afinidad = matriz.loc[p1, p2] if p1 in matriz.columns and p2 in matriz.columns else 0

                if afinidad > 0:
                    loc1 = individuo[p1]
                    loc2 = individuo[p2]
                    distancia = self.almacen._calcular_distancia_manhattan(loc1, loc2)
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
            if i < punto:
                hijo[producto] = padre1[producto]
            else:
                hijo[producto] = padre2[producto]

        # Corregir duplicados
        usadas = set(hijo.values())
        faltantes = [loc for loc in self.posibles_ubicaciones if loc not in usadas]

        ubicaciones_repetidas = [prod for prod, loc in hijo.items()
                                 if list(hijo.values()).count(loc) > 1]

        for producto in ubicaciones_repetidas:
            hijo[producto] = faltantes.pop()

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