import pandas as pd
import numpy as np
import random


class Almacen:
    def __init__(self, dataset_pth, affinity_pth, num_racks, lenght_rack, height_rack, num_productos_populares):
        print("Iniciando la creacion del modelo de almacen...")
        self.ancho = lenght_rack      # eje X
        # eje Y (los pares son estanterías, los impares pasillos)
        self.profundidad = num_racks * 2 - 1
        self.alto = height_rack       # eje Z
        self.x_corredor = 0           # Corredor principal en x=0
        self.punto_io = (self.x_corredor, self.profundidad //
                         2, 0)  # Punto medio del corredor

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
        self.datos_demanda = frecuencia.loc[productos_populares]

        affinity = pd.read_csv(affinity_fp, index_col=0)
        affinity.index = affinity.index.astype(str)
        affinity.columns = affinity.columns.astype(str)
        productos_populares = productos_populares.astype(str)
        affinity_filtrada = affinity.loc[productos_populares,
                                         productos_populares]

        return dataset[dataset['StockCode'].isin(productos_populares)], affinity_filtrada

    def _crear_ordenes_de_simulacion(self):
        ordenes = []
        productos_populares = self.datos_demanda.index.tolist()
        for _ in range(100):
            tamano_orden = np.random.randint(2, 6)
            orden = np.random.choice(
                productos_populares, size=tamano_orden, replace=False).tolist()
            ordenes.append(orden)
        return ordenes

    def _calcular_distancia_realista(self, punto1, punto2):
        """
        Calcula la distancia que recorrería un trabajador usando pasillos.
        """
        x1, y1, z1 = punto1
        x2, y2, z2 = punto2

        if y1 == y2:
            # estan en el mismo pasillo
            distancia = abs(x1 - x2) + abs(z1 - z2)
        else:
            # para cambiar de fila, hay que volver al corredor principal
            distancia = abs(x1 - self.x_corredor)  # salir del pasillo actual
            # caminar por el corredor principal
            distancia += abs(y1 - y2)
            distancia += abs(self.x_corredor - x2)  # entrar al nuevo pasillo
            distancia += abs(z1 - z2)             # ajustar altura
        return distancia

    def calcular_costo_total_simulacion(self):
        distancia_total = 0
        if not self.ubicaciones or len(self.ubicaciones) < len(self.datos_demanda.index):
            return float('inf')

        for orden in self.ordenes_simulacion:
            punto_actual = self.punto_io
            distancia_orden = 0
            # ordenar la ruta de picking dentro de la orden para ser mas eficiente
            # (Esto simula un trabajador que no recoge los productos al azar, sino de forma optimizada)
            sub_orden = sorted(orden, key=lambda p: self._calcular_distancia_realista(
                punto_actual, self.ubicaciones.get(p, self.punto_io)))

            for producto_id in sub_orden:
                if producto_id in self.ubicaciones:
                    destino = self.ubicaciones[producto_id]
                    distancia_orden += self._calcular_distancia_realista(
                        punto_actual, destino)
                    punto_actual = destino

            distancia_orden += self._calcular_distancia_realista(
                punto_actual, self.punto_io)
            distancia_total += distancia_orden

        return distancia_total / len(self.ordenes_simulacion)
