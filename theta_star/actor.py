import tensorflow as tf
from keras import layers
import numpy as np

class ResBlock(tf.keras.Model):
    def __init__(self, input_dim, output_dim, n_neurons=512):
        super(ResBlock, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.fc1 = layers.Dense(n_neurons, activation=None, kernel_initializer='he_uniform')
        # self.bn1 = layers.BatchNormalization()  
        self.activation = layers.LeakyReLU(negative_slope=0.2)

        self.fc2 = layers.Dense(output_dim, activation=None, kernel_initializer='he_uniform')
        # self.bn2 = layers.BatchNormalization()  

        if input_dim != output_dim:
            self.fc_res = layers.Dense(output_dim, activation=None, kernel_initializer='he_uniform')
        else:
            self.fc_res = None

    def call(self, x, final_nl=True):
        residual = x
        if self.fc_res:
            residual = self.fc_res(residual)
            residual = self.activation(residual)  

        x = self.fc1(x)
        # x = self.bn1(x)  
        x = self.activation(x)

        x = self.fc2(x)
        # x = self.bn2(x)  

        x += residual
        if final_nl:
            x = self.activation(x)
        return x


class ImprovedActor(tf.keras.Model):
    def __init__(self, state_dim, action_dim, n_neurons=512, trainable = None, dtype="float32"):
        super(ImprovedActor, self).__init__(dtype = dtype)
        # self.bn1 = layers.BatchNormalization()  
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.rb1 = ResBlock(state_dim, state_dim, n_neurons)
        self.rb2 = ResBlock(state_dim + state_dim, state_dim + state_dim, n_neurons)

        # Три выходных слоя с разными активациями
        self.out1 = layers.Dense(action_dim - 2, activation='sigmoid', kernel_initializer='he_uniform')
        self.out2 = layers.Dense(action_dim - 2, activation='tanh', kernel_initializer='he_uniform')
        self.out3 = layers.Dense(action_dim - 2, activation='relu', kernel_initializer='he_uniform')  # Для третьего действия
        # Dropout слой
        self.dropout = layers.Dropout(rate=0.1)

    def call(self, obs, training=True):
        if isinstance(obs, np.ndarray):
            obs = tf.convert_to_tensor(obs, dtype=tf.float32)

        x0 = obs
        x = self.rb1(x0, training=training)
        x = self.rb2(tf.concat([x0, x], axis=-1), training=training)

        # Получаем выходы для каждого действия
        output1 = self.out1(x)
        output2 = self.out2(x)
        output3 = self.out3(x)

        # Объединяем их
        prob = tf.concat([output1, output2, output3], axis=-1)

        # Нормализуем (сумма должна быть равна 1)
        prob = tf.nn.softmax(prob, axis=-1)

        return prob
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "n_neurons": 512
        })
        print("СОХРАНЯЕМ CONFIG:", config)  # Отладка
        return config

    @classmethod
    def from_config(cls, config):
        print("ЗАГРУЖАЕМ CONFIG:", config)  # Отладка
        return cls(
            state_dim=config.get("state_dim", 4),  
            action_dim=config.get("action_dim", 3),  
            n_neurons=config.get("n_neurons", 512),
            dtype=config.get("dtype", "float32")
        )