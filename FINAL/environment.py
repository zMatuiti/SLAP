import pandas as pd
import numpy as np


class Almacen:
    def __init__(self, dataset_pth, affinity_pth, num_racks, lenght_rack, height_rack, num_productos_populares):
        print("Iniciando la creacion del modelo de almacen...")

        # estructura fisica del almacen
        self.ancho = lenght_rack      # eje X
        self.profundidad = num_racks  # eje Y
        self.alto = height_rack       # eje Z
        self.total_spaces = self.ancho * self.profundidad * self.alto
        # punto de entrada/salida para el picking
        self.punto_io = (0, 0, 0)

        # diccionario para guardar asignaciones: {'producto_id': (x, y, z)}
        self.ubicaciones = {}

        self.dataset, self.affinity = self._load_dataset_and_affinity(
            dataset_pth, affinity_pth, num_productos_populares)
        print(
            f"\nSe analizaron los {num_productos_populares} productos mas populares.")

        self.ordenes_simulacion = self._crear_ordenes_de_simulacion()
        print(
            f"Se crearon {len(self.ordenes_simulacion)} ordenes de simulacion para las pruebas.")

    def _load_dataset_and_affinity(self, dataset_fp, affinity_fp, num_productos):
        dataset = pd.read_csv(dataset_fp)
        frecuencia = dataset['StockCode'].value_counts()
        productos_populares = frecuencia.head(num_productos).index
        dataset_filtrado = dataset[dataset['StockCode'].isin(
            productos_populares)]
        self.datos_demanda = frecuencia.loc[productos_populares]

        affinity = pd.read_csv(affinity_fp, index_col=0)
        affinity.index = affinity.index.astype(str)
        affinity.columns = affinity.columns.astype(str)
        productos_populares = productos_populares.astype(str)
        affinity_filtrada = affinity.loc[productos_populares,
                                         productos_populares]

        return dataset_filtrado, affinity_filtrada

    def _crear_ordenes_de_simulacion(self):
        ordenes = []
        productos_populares = self.datos_demanda.index.tolist()
        for _ in range(100):
            tamano_orden = np.random.randint(2, 6)
            orden = np.random.choice(
                productos_populares, size=tamano_orden, replace=False).tolist()
            ordenes.append(orden)
        return ordenes

    def _calcular_distancia_manhattan(self, punto1, punto2):
        return sum(abs(p1 - p2) for p1, p2 in zip(punto1, punto2))

    def asignar_producto(self, producto_id, coordenadas):
        x, y, z = coordenadas
        if (0 <= x < self.ancho) and (0 <= y < self.profundidad) and (0 <= z < self.alto):
            self.ubicaciones[producto_id] = coordenadas
        else:
            print(f"Error: Coordenadas {coordenadas} fuera de los limites.")

    def calcular_costo_total_simulacion(self):
        distancia_total = 0
        if not self.ubicaciones or len(self.ubicaciones) < len(self.datos_demanda.index):
            return float('inf')

        for orden in self.ordenes_simulacion:
            punto_actual = self.punto_io
            distancia_orden = 0
            for producto_id in orden:
                if producto_id in self.ubicaciones:
                    destino = self.ubicaciones[producto_id]
                    distancia_orden += self._calcular_distancia_manhattan(
                        punto_actual, destino)
                    punto_actual = destino
            distancia_orden += self._calcular_distancia_manhattan(
                punto_actual, self.punto_io)
            distancia_total += distancia_orden
        return distancia_total / len(self.ordenes_simulacion)
