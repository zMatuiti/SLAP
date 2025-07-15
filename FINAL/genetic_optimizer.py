import random
from environment import Almacen


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
                f"No hay suficientes ubicaciones ({len(self.posibles_ubicaciones)}) para los productos ({len(self.productos)})."
            )

    def _generar_posibles_ubicaciones(self):
        # genera ubicaciones solo en las estanterias (coordenadas Y pares)
        ubicaciones = []
        for x in range(self.almacen.ancho):
            for y in range(0, self.almacen.profundidad, 2):
                for z in range(self.almacen.alto):
                    ubicaciones.append((x, y, z))
        return ubicaciones

    def _crear_individuo(self):
        ubicaciones = random.sample(
            self.posibles_ubicaciones, len(self.productos))
        return dict(zip(self.productos, ubicaciones))

    def _fitness(self, individuo):
        self.almacen.ubicaciones = individuo
        costo_picking = self.almacen.calcular_costo_total_simulacion()

        costo_afinidad = 0
        matriz = self.almacen.affinity
        productos = self.productos
        for i in range(len(productos)):
            for j in range(i + 1, len(productos)):
                p1, p2 = productos[i], productos[j]
                if p1 in matriz.index and p2 in matriz.columns:
                    afinidad = matriz.loc[p1, p2]
                    if afinidad > 0:
                        loc1 = individuo[p1]
                        loc2 = individuo[p2]
                        distancia = self.almacen._calcular_distancia_realista(
                            loc1, loc2)
                        costo_afinidad += afinidad * distancia

        if len(productos) > 1:
            costo_afinidad /= (len(productos) * (len(productos) - 1) / 2)

        # combina ambos costos.
        # se pueden ajustar los pesos alpha y beta.
        return 0.7 * costo_picking + 0.3 * costo_afinidad

    def _seleccionar_padres(self, poblacion, scores):
        padres = []
        for _ in range(2):
            torneo = random.sample(list(zip(poblacion, scores)), k=5)
            torneo.sort(key=lambda x: x[1])  # menor costo = mejor
            padres.append(torneo[0][0])
        return padres

    def _crossover(self, padre1, padre2):
        hijo = {}
        punto_inicio = random.randint(0, len(self.productos) - 2)
        punto_fin = random.randint(punto_inicio + 1, len(self.productos) - 1)

        genes_padre1 = list(padre1.items())
        hijo_genes_iniciales = dict(genes_padre1[punto_inicio:punto_fin])
        hijo = hijo_genes_iniciales.copy()

        ubicaciones_usadas = set(hijo.values())
        for producto, ubicacion in padre2.items():
            if producto not in hijo and ubicacion not in ubicaciones_usadas:
                hijo[producto] = ubicacion
                ubicaciones_usadas.add(ubicacion)

        if len(hijo) < len(self.productos):
            productos_faltantes = [p for p in self.productos if p not in hijo]
            ubicaciones_libres = [
                loc for loc in self.posibles_ubicaciones if loc not in hijo.values()]
            random.shuffle(ubicaciones_libres)

            for producto in productos_faltantes:
                if ubicaciones_libres:
                    hijo[producto] = ubicaciones_libres.pop()

        return hijo

    def _mutar(self, individuo):
        if random.random() < self.prob_mutacion:
            p1, p2 = random.sample(self.productos, 2)
            individuo[p1], individuo[p2] = individuo[p2], individuo[p1]
        return individuo

    def optimizar(self):
        poblacion = [self._crear_individuo()
                     for _ in range(self.poblacion_size)]
        mejor_individuo_global = None
        mejor_score_global = float('inf')

        historial_costos = []

        for gen in range(self.generaciones):
            scores = [self._fitness(ind) for ind in poblacion]
            mejor_score_gen = min(scores)

            historial_costos.append(mejor_score_gen)

            if mejor_score_gen < mejor_score_global:
                mejor_score_global = mejor_score_gen
                mejor_individuo_global = poblacion[scores.index(
                    mejor_score_gen)]

            print(f"Generacion {gen+1}: Mejor Costo = {mejor_score_gen:.2f}")

            nueva_poblacion = [mejor_individuo_global]

            while len(nueva_poblacion) < self.poblacion_size:
                padre1, padre2 = self._seleccionar_padres(poblacion, scores)
                hijo = self._crossover(padre1, padre2)
                hijo = self._mutar(hijo)
                nueva_poblacion.append(hijo)

            poblacion = nueva_poblacion

        print(f"\nMejor Costo Encontrado por GA: {mejor_score_global:.2f}")

        return mejor_individuo_global, historial_costos
