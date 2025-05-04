import tensorflow as tf
from tensorflow import keras
from keras import layers
import numpy as np
import matplotlib.pyplot as plt

def world_to_map(world_coords, resolution, origin, map_offset, map_shape):
    x_world, y_world = world_coords
    if isinstance(x_world, tf.Tensor):
        x_world = x_world.numpy()
    if isinstance(y_world, tf.Tensor):
        y_world = y_world.numpy()

    # Проверка на NaN и Inf
    if isinstance(x_world, np.ndarray):
        mask = np.isfinite(x_world) & np.isfinite(y_world)
        x_world, y_world = x_world[mask], y_world[mask]

    elif not np.isfinite(x_world) or not np.isfinite(y_world):
        return None, None  # Вернем None, если координаты некорректны

    # Преобразование
    x_map = ((x_world - origin[0]) / resolution).astype(int) + map_offset[0]
    y_map = ((y_world - origin[1]) / resolution).astype(int) + map_offset[1]

    # Переворот Y
    y_map = map_shape[0] - y_map - 1

    # Ограничение координат
    x_map = np.clip(x_map, 0, map_shape[1] - 1)
    y_map = np.clip(y_map, 0, map_shape[0] - 1)

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

    def call(self, obs, training=False):
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
        
        x_norm = tf.identity(obs)

        x0 = x_norm
        x = self.rb1(x0, final_nl=True)
        x = self.rb2(tf.concat([x0, x], axis=-1), final_nl=True)
        x = self.dropout(x, training=training)
        output = self.out(x)
   
        return output
    

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
        x_map, y_map = world_to_map((x, y), resolution=0.05, origin=(-4.86, -7.36),
                                    map_offset=(45, 15), map_shape=self.grid_map.shape)
        return self.value_map[y_map, x_map]  # Достаём значение из таблицы
    

    