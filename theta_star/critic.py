import tensorflow as tf
# from tensorflow import kears
from keras import layers
import numpy as np
import matplotlib.pyplot as plt

def world_to_map(world_coords, resolution, origin, map_offset, map_shape):
    x_world, y_world = world_coords

    # Для одиночных значений (не массивов)
    if not isinstance(x_world, np.ndarray):
        x_map = int((x_world - origin[0]) / resolution) + map_offset[0]
        y_map = int((y_world - origin[1]) / resolution) + map_offset[1]
    else:
        # Для массивов
        x_map = ((x_world - origin[0]) / resolution).astype(int) + map_offset[0]
        y_map = ((y_world - origin[1]) / resolution).astype(int) + map_offset[1]

    # # Переворот координаты Y
    # y_map = map_shape[0] - y_map - 1

    # # Ограничение диапазона
    # x_map = np.clip(x_map, 0, map_shape[1]-1) if isinstance(x_map, np.ndarray) else max(0, min(x_map, map_shape[1]-1))
    # y_map = np.clip(y_map, 0, map_shape[0]-1) if isinstance(y_map, np.ndarray) else max(0, min(y_map, map_shape[0]-1))

    return x_map, y_map

# -------------------------------
# Определение блока ResBlock
# -------------------------------
class ResBlock(tf.keras.Model):
    def __init__(self, input_dim, output_dim, n_neurons=512):
        super(ResBlock, self).__init__()
        self.fc1 = layers.Dense(n_neurons, activation=None, kernel_initializer='he_uniform')
        self.activation = layers.LeakyReLU(negative_slope=0.2)
        self.fc2 = layers.Dense(output_dim, activation=None, kernel_initializer='he_uniform')

        # Если размерности не совпадают, создаём слой для выравнивания размерностей остатка
        if input_dim != output_dim:
            self.fc_res = layers.Dense(output_dim, activation=None, kernel_initializer='he_uniform')
        else:
            self.fc_res = None

    def call(self, x, final_nl=True):
        residual = x
        if self.fc_res:
            residual = self.fc_res(residual)
            # residual = self.activation(residual)  # Применяем активацию к выравниванию

        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        x += residual
        if final_nl:
            x = self.activation(x)
        return x


# -------------------------------
# Определение улучшенного критика
# -------------------------------
class ImprovedCritic(tf.keras.Model):
    def __init__(self, state_dim, grid_map, optimal_path, n_neurons=512):
        """
        :param state_dim: размерность вектора состояния (без учета дополнительного признака)
        :param grid_map: numpy-массив карты (например, 2D массив, где 1 – препятствие)
        :param optimal_path: список точек оптимального пути, например [(x1, y1), (x2, y2), ...]
        :param n_neurons: количество нейронов во внутренних слоях
        """
        super(ImprovedCritic, self).__init__()
        self.grid_map = grid_map
        self.optimal_path = optimal_path
        # self.value_map = None
        # self.dev_mean = 1.0
        # self.dev_std = 1.0
        # self.deviation_list = []
        # self.penality_list = []
        # self.history = []
        # self.max_history = 1000

        # На вход подаем состояние плюс один дополнительный признак (отклонение + штраф)
        self.rb1 = ResBlock(state_dim , state_dim, n_neurons)
        self.rb2 = ResBlock((state_dim) * 2, (state_dim) * 2, n_neurons)
        self.dropout = layers.Dropout(rate=0.1)
        self.out = layers.Dense(1, activation=None, kernel_initializer='he_uniform')

    def call(self, obs, training=True):
        """
        :param obs: вектор состояния, например [x, y, ...]
        :param deviation_from_path: отклонение от оптимального пути (скаляр)
        :param collision_penalty: штраф за столкновение (скаляр)
        :param training: режим обучения (для Dropout)
        :return: оценка ценности состояния
        """
        # Приводим входные данные к тензорам, если они заданы как numpy или числа
        if isinstance(obs, np.ndarray):
            obs = tf.convert_to_tensor(obs, dtype=tf.float32)

        if tf.reduce_any(tf.math.is_nan(obs)) or tf.reduce_any(tf.math.is_inf(obs)):
            tf.print("Обнаружены некорректные значения (NaN или Inf) в combined_deviation:", obs)
            tf.debugging.check_numerics(obs, "Проверка obs_with_deviation")
            return tf.constant([[float(0)]])  # Возвращаем безопасное значение, если обнаружены NaN или Inf
        
        #Если obs не батчевый (например, имеет форму (state_dim,)), добавляем размерность батча
        if len(obs.shape) == 1:
            obs = tf.expand_dims(obs, axis=0)

        # Проход через резидуальные блоки
        x0 = obs
        x = self.rb1(x0, final_nl=True)
        x = self.rb2(tf.concat([x0, x], axis=-1), final_nl=True)
        x = self.dropout(x, training=training)
        output = self.out(x)
        return output
    
    # def eval_value(self, state, grid_map):
    #     """
    #     Оценивает значение состояния (работает с первым образцом, если state – батч)
    #     :param state: вектор состояния или батч состояний
    #     :return: оценка ценности
    #     """
        
    #     # print(len(state))
    #     # Если state является батчем, берём первый элемент
    #     if state.ndim == 2 and state.shape[0] == 1:
    #         state = state[0]

        
    #     state_sample = [state[0], state[1]]
    #     st = [state[2], state[3]]
    #     state_sample = world_to_map(state_sample, resolution = 0.05, origin = (-4.86, -7.36),  map_offset = (45, 15), map_shape = grid_map.shape)
        
    #     # Предполагаем, что первые два элемента вектора – координаты (x, y)
    #     # Приводим их к числам для работы с numpy
    #     current_pos = (float(state_sample[0].numpy()) if hasattr(state_sample[0], 'numpy') else float(state_sample[0]),
    #                    float(state_sample[1].numpy()) if hasattr(state_sample[1], 'numpy') else float(state_sample[1]))
    #     deviation = compute_deviation_from_path(current_pos, self.optimal_path)
    #     self.deviation_list.append(deviation)
    #     # Определяем штраф за столкновение
    #     collision_penalty = 0
    #     if self.is_near_obstacle(current_pos):
    #         assert isinstance(collision_penalty, (int, float))
    #         collision_penalty = 10 # Значительный штраф за приближение к препятствию
    #     self.penality_list.append(collision_penalty)
    #     state = tf.concat([
    #     tf.convert_to_tensor(state_sample, dtype=tf.float32),  # Пиксельные координаты
    #     tf.convert_to_tensor(st, dtype=tf.float32)             # Доп. параметры
    #     ], axis=-1)

    #     # print(state)
    #     return self.call(state, deviation, collision_penalty)

    
    # def is_near_obstacle(self, point, safe_distance=2):
    #     """
    #     Проверяет, находится ли точка рядом с препятствием на карте.
    #     :param point: координаты точки (x, y)
    #     :param safe_distance: радиус проверки вокруг точки
    #     :return: True, если в области обнаружено препятствие
    #     """
    #     x, y = int(round(point[0])), int(round(point[1]))
    #     h, w = self.grid_map.shape

    #     if x < 0 or y < 0 or x >= w or y >= h:
    #         return True

    #     # Определяем границы области проверки (учитываем, что срез в numpy не включает правую границу)
    #     x_min = max(0, x - safe_distance)
    #     x_max = min(w, x + safe_distance + 1)
    #     y_min = max(0, y - safe_distance)
    #     y_max = min(h, y + safe_distance + 1)

    #     area = self.grid_map[y_min:y_max, x_min:x_max]
    #     return np.any(area == 1)


class StaticCritic:
    def __init__(self, value_map, grid_map):
        self.value_map = value_map
        self.grid_map = grid_map

    def call(self, state):
        """ Получаем значение критика из предвычисленной карты """
        if isinstance(state, tf.Tensor):
            state = state.numpy()
        if state.ndim == 2 and state.shape[0] == 1:
            state = state[0] 
        x, y = state[:2]  # Берём координаты состояния
        x_map, y_map = world_to_map((x, y), resolution=0.05, origin=(-7.76, -7.15),
                                    map_offset=(0, 0), map_shape=self.grid_map.shape)
        return self.value_map[y_map, x_map]  # Достаём значение из таблицы
    

    