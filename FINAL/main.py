from environment import Almacen
from genetic_optimizer import GeneticOptimizer
from dqn_agent import train_dqn
import random
import matplotlib.pyplot as plt
import numpy as np

# semilla para resultados reproducibles
random.seed(42)
np.random.seed(42)


if __name__ == "__main__":
    data_file = 'OnlineRetail_Cleaned.csv'
    affinity_matrix_file = 'affinity_matrix.csv'

    # parametros del almacen y productos a analizar
    NUM_PRODUCTOS = 15
    NUM_GENERACIONES_GA = 50
    NUM_EPISODIOS_DQN = 100

    mi_almacen = Almacen(data_file, affinity_matrix_file,
                         num_racks=5,
                         lenght_rack=5,
                         height_rack=2,
                         num_productos_populares=NUM_PRODUCTOS)

    print("\n--- Ejecutando Baseline Aleatorio ---")
    # para una comparacion 100% justa, usamos el mismo metodo para generar ubicaciones
    posibles_ubicaciones = GeneticOptimizer(
        mi_almacen)._generar_posibles_ubicaciones()
    productos = mi_almacen.datos_demanda.index.tolist()

    ubicaciones_aleatorias = random.sample(
        posibles_ubicaciones, len(productos))
    asignacion_aleatoria = dict(zip(productos, ubicaciones_aleatorias))

    mi_almacen.ubicaciones = asignacion_aleatoria
    costo_baseline = mi_almacen.calcular_costo_total_simulacion()
    print(f"Costo del Baseline Aleatorio: {costo_baseline:.2f}")

    # --- ALGORITMO GENETICO ---
    print("\n--- Ejecutando Algoritmo Genetico ---")
    optimizador_ga = GeneticOptimizer(
        mi_almacen, poblacion=50, generaciones=NUM_GENERACIONES_GA, prob_mutacion=0.1)

    # captura la asignacion Y el historial
    mejor_asignacion_ga, historial_ga = optimizador_ga.optimizar()

    mi_almacen.ubicaciones = mejor_asignacion_ga
    costo_genetico = mi_almacen.calcular_costo_total_simulacion()
    print(f"Costo final tras optimizacion con GA: {costo_genetico:.2f}")

    # --- AGENTE DQN ---
    print("\n--- Ejecutando Agente DQN ---")

    # captura los dos resultados (asignacion e historial)
    asignacion_dqn, historial_dqn = train_dqn(
        mi_almacen, episodios=NUM_EPISODIOS_DQN)

    mi_almacen.ubicaciones = asignacion_dqn
    costo_dqn = mi_almacen.calcular_costo_total_simulacion()
    print(f"Costo final tras optimizacion con DQN: {costo_dqn:.2f}")

    # --- COMPARACION FINAL ---
    print("\n\n--- COMPARACIoN DE RESULTADOS ---")
    print(f"Costo Baseline Aleatorio: {costo_baseline:.2f}")
    print(f"Costo Algoritmo Genetico: {costo_genetico:.2f}")
    print(f"Costo DQN: {costo_dqn:.2f}")

    # --- GRAFICOS DE RESULTADOS ---

    # grafico de barras
    metodos = ['Aleatorio', 'Algoritmo Genetico', 'DQN']
    costos = [costo_baseline, costo_genetico, costo_dqn]

    plt.figure(figsize=(10, 6))
    plt.bar(metodos, costos, color=['gray', 'blue', 'green'])
    plt.title('Comparacion de Metodos de Optimizacion de Almacen')
    plt.ylabel('Costo Promedio de Picking (Distancia)')
    plt.show()

    # --- graficos de las curvas de aprendizaje ---

    # grafico de Convergencia del Algoritmo Genetico
    plt.figure(figsize=(10, 6))
    plt.plot(historial_ga)
    plt.title('Convergencia del Algoritmo Genetico')
    plt.xlabel('Generacion')
    plt.ylabel('Mejor Costo de Fitness')
    plt.grid(True)
    plt.show()

    # Grafico de Aprendizaje del Agente DQN
    plt.figure(figsize=(10, 6))
    plt.plot(historial_dqn)
    plt.title('Curva de Aprendizaje del Agente DQN')
    plt.xlabel('Episodio')
    plt.ylabel('Costo de Picking del Episodio')
    plt.grid(True)
    plt.show()
