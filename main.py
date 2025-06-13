import pandas as pd
import numpy as np

class Almacen:
    def __init__(self, archivo_datos, ancho_almacen=20, alto_almacen=10, num_productos_populares=50):
        print("Iniciando la creacion del modelo de almacen...")
        
        # estructura fisica del almacen
        self.ancho = ancho_almacen
        self.alto = alto_almacen
        self.total_ubicaciones = ancho_almacen * alto_almacen
        self.punto_io = (0, 0) # Punto de entrada/salida para el picking
        
        # diccionario para guardar las asignaciones: {'producto_id': (x, y)}
        self.ubicaciones = {}
        
        # cargar y procesar los datos para entender la demanda 
        self.datos_demanda = self._cargar_y_procesar_demanda(archivo_datos, num_productos_populares)
        print(f"\nSe analizaron los {num_productos_populares} productos mas populares.")
        
        # lista de ordenes para simular y validar el costo de picking
        self.ordenes_simulacion = self._crear_ordenes_de_simulacion()
        print(f"Se crearon {len(self.ordenes_simulacion)} ordenes de simulacion para las pruebas.")

    def _cargar_y_procesar_demanda(self, filepath, num_productos):
        # carga el archivo CSV, lo limpia y calcula la frecuencia (demanda) de cada producto.
        print(f"\nCargando datos desde: {filepath}...")
        df = pd.read_csv(filepath)
        
        # limpieza de datos
        df.dropna(subset=['Invoice', 'StockCode', 'Customer ID'], inplace=True)
        df = df[~df['Invoice'].astype(str).str.startswith('C')] # eliminar devoluciones
        df = df[df['Quantity'] > 0]
        
        print("Calculando la demanda de productos (rotacion)...")
        # contar la frecuencia de cada producto
        frecuencia_productos = df['StockCode'].value_counts()
        
        # devolvemos solo los n productos mas populares
        return frecuencia_productos.head(num_productos)

    def _crear_ordenes_de_simulacion(self):
        # se crea una lista de ordenes de ejemplo usando los productos mas populares.
        # cada orden es una lista de IDs de productos.
        ordenes = []
        productos_populares = self.datos_demanda.index.tolist()
        
        for _ in range(100): # Crear 100 ordenes de prueba
            tamano_orden = np.random.randint(2, 6) # ordenes con 2 a 5 productos
            orden = np.random.choice(productos_populares, size=tamano_orden, replace=False).tolist()
            ordenes.append(orden)
        return ordenes

    def _calcular_distancia_manhattan(self, punto1, punto2):
        # se calcula la distancia de Manhattan, ideal para un layout de almacen en cuadricula.
        return abs(punto1[0] - punto2[0]) + abs(punto1[1] - punto2[1])

    def asignar_producto(self, producto_id, coordenadas):
        # asigna o reasigna un producto a una nueva ubicacion en el almacen.
        # esta seria una 'accion' de nuestro agente DQN.
        if 0 <= coordenadas[0] < self.ancho and 0 <= coordenadas[1] < self.alto:
            self.ubicaciones[producto_id] = coordenadas
        else:
            print(f"Error: Coordenadas {coordenadas} fuera de los limites del almacen.")

    def calcular_costo_total_simulacion(self):
        # calcula la distancia total de picking para todas las ordenes de simulacion.
        # este es el valor que queremos minimizar.
        
        distancia_total = 0
        if not self.ubicaciones:
            return float('inf') # si no hay productos asignados, el costo es infinito

        for orden in self.ordenes_simulacion:
            punto_actual = self.punto_io
            distancia_orden = 0
            
            for producto_id in orden:
                if producto_id in self.ubicaciones:
                    destino = self.ubicaciones[producto_id]
                    distancia_orden += self._calcular_distancia_manhattan(punto_actual, destino)
                    punto_actual = destino
            
            # sumar la distancia de regreso al punto de inicio
            distancia_orden += self._calcular_distancia_manhattan(punto_actual, self.punto_io)
            distancia_total += distancia_orden
            
        return distancia_total / len(self.ordenes_simulacion) # devolvemos el costo promedio por orden

if __name__ == "__main__":
    
    # ruta al archivo que descargaste
    archivo_de_datos = 'online_retail_II.csv'
    
    # se creaa el entorno del almacen
    mi_almacen = Almacen(archivo_de_datos)
    
    print("\n--- Demostracion de asignacion aleatoria ---")
    
    # asigna los productos populares a ubicaciones aleatorias
    productos_a_asignar = mi_almacen.datos_demanda.index.tolist()
    
    for producto in productos_a_asignar:
        x_aleatorio = np.random.randint(0, mi_almacen.ancho)
        y_aleatorio = np.random.randint(0, mi_almacen.alto)
        mi_almacen.asignar_producto(producto, (x_aleatorio, y_aleatorio))
        
    print(f"\nSe asignaron {len(mi_almacen.ubicaciones)} productos a ubicaciones aleatorias.")
    
    # se calcula el costo total de esta asignacion aleatoria
    costo_promedio = mi_almacen.calcular_costo_total_simulacion()
    
    print(f"\nCosto Promedio de Picking (Distancia por Orden): {costo_promedio:.2f}")
    