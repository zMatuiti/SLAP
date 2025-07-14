from environment import Almacen
from genetic_optimizer import GeneticOptimizer
from dqn_agent import train_dqn
import random
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # configuracion
    data_file = 'OnlineRetail_Cleaned.csv'
    affinity_matrix_file = 'affinity_matrix.csv'

    # parametros del almacen y productos a analizar
    NUM_PRODUCTOS = 15  # reducir para que el DQN entrene mas rapido

    mi_almacen = Almacen(data_file, affinity_matrix_file,
                         num_racks=5,
                         lenght_rack=5,
                         height_rack=2,
                         num_productos_populares=NUM_PRODUCTOS)

    # baseline aleatorio
    print("\n Ejecutando Baseline Aleatorio ")
    productos = mi_almacen.datos_demanda.index.tolist()
    posibles_ubicaciones = []
    for x in range(mi_almacen.ancho):
        for y in range(mi_almacen.profundidad):
            for z in range(mi_almacen.alto):
                posibles_ubicaciones.append((x, y, z))

    ubicaciones_aleatorias = random.sample(
        posibles_ubicaciones, len(productos))
    asignacion_aleatoria = dict(zip(productos, ubicaciones_aleatorias))

    mi_almacen.ubicaciones = asignacion_aleatoria
    costo_baseline = mi_almacen.calcular_costo_total_simulacion()
    print(f"Costo del Baseline Aleatorio: {costo_baseline:.2f}")

    # algoritmo genetico
    print("\n Ejecutando Algoritmo Genetico ")
    optimizador_ga = GeneticOptimizer(
        mi_almacen, poblacion=50, generaciones=50, prob_mutacion=0.1)
    mejor_asignacion_ga = optimizador_ga.optimizar()
    mi_almacen.ubicaciones = mejor_asignacion_ga
    costo_genetico = mi_almacen.calcular_costo_total_simulacion()
    print(f"Costo final tras optimizacion con GA: {costo_genetico:.2f}")

    # agente DQN 
    print("\n Ejecutando Agente DQN ")
    # Descomenta las siguientes lineas y borra la del placeholder
    asignacion_dqn = train_dqn(mi_almacen, episodios=100)
    mi_almacen.ubicaciones = asignacion_dqn
    costo_dqn = mi_almacen.calcular_costo_total_simulacion()
    print(f"Costo final tras optimizacion con DQN: {costo_dqn:.2f}")

    # comparacion final 
    print("\n\n COMPARACIoN DE RESULTADOS ")
    print(f"Costo Baseline Aleatorio: {costo_baseline:.2f}")
    print(f"Costo Algoritmo Genetico: {costo_genetico:.2f}")
    print(f"Costo DQN: {costo_dqn:.2f}")

    # crea el grafico comparativo 
    metodos = ['Aleatorio', 'Algoritmo Genetico', 'DQN']
    costos = [costo_baseline, costo_genetico, costo_dqn]

    plt.figure(figsize=(10, 6))
    plt.bar(metodos, costos, color=['gray', 'blue', 'green'])
    plt.title('Comparacion de Metodos de Optimizacion de Almacen')
    plt.ylabel('Costo Promedio de Picking (Distancia)')
    plt.show()
