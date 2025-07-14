import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Lambda
from tensorflow.keras.optimizers import Adam
from collections import deque
import numpy as np
import random
from environment import Almacen


class DQNAgent:
    def __init__(self, almacen):
        self.almacen = almacen
        self.productos_a_colocar = almacen.datos_demanda.index.tolist()
        self.posibles_ubicaciones = self._generar_posibles_ubicaciones()
        self.num_productos = len(self.productos_a_colocar)
        
        self.state_size = self.num_productos + 3  # one-hot-prod + x,y,z
        self.action_size = len(self.posibles_ubicaciones)
        self.memory = deque(maxlen=20000)

        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001

        self.model = self._build_dueling_dqn_model()
        self.target_model = self._build_dueling_dqn_model()
        self.update_target_model()

    def _generar_posibles_ubicaciones(self):
        ubicaciones = []
        for x in range(self.almacen.ancho):
            for y in range(self.almacen.profundidad):
                for z in range(self.almacen.alto):
                    ubicaciones.append((x, y, z))
        return ubicaciones

    def _build_dueling_dqn_model(self):
        state_input = Input(shape=(self.state_size,))
        dense1 = Dense(128, activation='relu')(state_input)
        dense2 = Dense(128, activation='relu')(dense1)
        value_stream = Dense(64, activation='relu')(dense2)
        value = Dense(1, activation='linear')(value_stream)
        advantage_stream = Dense(64, activation='relu')(dense2)
        advantage = Dense(self.action_size, activation='linear')(advantage_stream)
        
        def dueling_q_formula(args):
            v_stream, a_stream = args
            # formula: Q = V + (A - mean(A))
            return v_stream + (a_stream - tf.reduce_mean(a_stream, axis=1, keepdims=True))

        q_values = Lambda(dueling_q_formula, output_shape=(self.action_size,))([value, advantage])
        
        model = Model(inputs=state_input, outputs=q_values)
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, invalid_actions_mask):
        if np.random.rand() <= self.epsilon:
            # para la exploracion aleatoria, tambien debemos evitar acciones invalidas
            valid_actions = np.where(invalid_actions_mask == False)[0]
            if len(valid_actions) == 0:
                # si no hay acciones validas, devuelve una accion aleatoria
                return random.randrange(self.action_size) 
            return np.random.choice(valid_actions)

        state = np.array([state])
        q_values = self.model.predict(state, verbose=0)[0]
        q_values[invalid_actions_mask] = -np.inf  # aplica la mascara
        return np.argmax(q_values)

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        
        states = np.array([t[0] for t in minibatch])
        next_states = np.array([t[3] for t in minibatch])
        
        q_values_current = self.model.predict(states, verbose=0)
        q_values_next = self.target_model.predict(next_states, verbose=0)

        for i, (state, action, reward, next_state, done) in enumerate(minibatch):
            if done:
                target_q = reward
            else:
                target_q = reward + self.gamma * np.amax(q_values_next[i])
            q_values_current[i][action] = target_q
            
        self.model.fit(states, q_values_current, epochs=1, verbose=0)

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


def train_dqn(almacen, episodios=100, batch_size=32):
    agent = DQNAgent(almacen)
    
    for e in range(episodios):
        productos_a_colocar = list(agent.productos_a_colocar)
        random.shuffle(productos_a_colocar)
        
        ultima_ubicacion = (0, 0, 0)
        episodio_memoria = []
        asignacion_actual = {}

        # ejecuta un episodio completo
        for i, producto_actual in enumerate(productos_a_colocar):
            prod_idx = agent.productos_a_colocar.index(producto_actual)
            prod_one_hot = tf.keras.utils.to_categorical(prod_idx, num_classes=agent.num_productos)
            current_state = np.concatenate([prod_one_hot, np.array(ultima_ubicacion)])

            # crea la mascara de acciones invalidas
            invalid_action_indices = [agent.posibles_ubicaciones.index(loc) for loc in asignacion_actual.values()]
            mask = np.zeros(agent.action_size, dtype=bool)
            mask[invalid_action_indices] = True

            # el agente elige una accion valida
            action = agent.act(current_state, invalid_actions_mask=mask)
            
            ubicacion_seleccionada = agent.posibles_ubicaciones[action]
            asignacion_actual[producto_actual] = ubicacion_seleccionada

            done = (i == len(productos_a_colocar) - 1)
            
            # crea el siguiente estado
            if done:
                next_state = np.zeros(agent.state_size)
            else:
                siguiente_prod = productos_a_colocar[i + 1]
                siguiente_prod_idx = agent.productos_a_colocar.index(siguiente_prod)
                siguiente_prod_one_hot = tf.keras.utils.to_categorical(siguiente_prod_idx, num_classes=agent.num_productos)
                next_state = np.concatenate([siguiente_prod_one_hot, np.array(ubicacion_seleccionada)])
            
            episodio_memoria.append((current_state, action, 0, next_state, done))
            ultima_ubicacion = ubicacion_seleccionada

        # calcula recompensa final para el episodio
        almacen.ubicaciones = asignacion_actual
        costo_picking = almacen.calcular_costo_total_simulacion()
        
        costo_afinidad = 0
        matriz_afinidad = almacen.affinity
        for i in range(len(productos_a_colocar)):
            for j in range(i + 1, len(productos_a_colocar)):
                p1, p2 = productos_a_colocar[i], productos_a_colocar[j]
                afinidad = matriz_afinidad.loc[p1, p2]
                distancia = almacen._calcular_distancia_manhattan(asignacion_actual[p1], asignacion_actual[p2])
                costo_afinidad += afinidad * distancia
        
        costo_total = 0.7 * costo_picking + 0.3 * costo_afinidad
        recompensa_final = -costo_total

        # actualiza la memoria con la recompensa real y entrenar
        for state, action, _, next_state, done in episodio_memoria:
            agent.remember(state, action, recompensa_final, next_state, done)
        
        if len(agent.memory) > batch_size:
            agent.replay(batch_size)
        
        if e % 10 == 0: # actualizaa el target model cada 10 episodios
            agent.update_target_model()
            
        agent.decay_epsilon()
        
        print(f"Episodio {e+1}/{episodios} | Costo Total: {costo_total:.2f} | Epsilon: {agent.epsilon:.2f}")
    
    # devuelve la mejor asignacion encontrada
    print("\nEntrenamiento DQN finalizado. Generando asignacion optima...")
    agente_entrenado = agent
    agente_entrenado.epsilon = 0.0  # apagar exploracion
    
    asignacion_final = {}
    productos_finales = list(agente_entrenado.productos_a_colocar)
    productos_finales.sort() # ordenaa la evaluacion para que sea consistente
    ultima_pos = (0, 0, 0)

    for producto in productos_finales:
        prod_idx = agente_entrenado.productos_a_colocar.index(producto)
        prod_one_hot = tf.keras.utils.to_categorical(prod_idx, num_classes=agent.num_productos)
        estado = np.concatenate([prod_one_hot, np.array(ultima_pos)])
        
        # crea la mascara para la asignacion final
        invalid_action_indices = [agente_entrenado.posibles_ubicaciones.index(loc) for loc in asignacion_final.values()]
        mask = np.zeros(agente_entrenado.action_size, dtype=bool)
        mask[invalid_action_indices] = True

        accion = agente_entrenado.act(estado, invalid_actions_mask=mask)
        
        ubicacion = agente_entrenado.posibles_ubicaciones[accion]
        asignacion_final[producto] = ubicacion
        ultima_pos = ubicacion

    return asignacion_final
