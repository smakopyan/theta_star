import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow import keras
from keras import layers
from keras import initializers
import numpy as np

class ResBlock(tf.keras.Model):
    def __init__(self, input_dim, output_dim, n_neurons=512):
        super(ResBlock, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.fc1 = layers.Dense(n_neurons, activation=None, kernel_initializer='he_uniform')
        self.activation = layers.LeakyReLU(negative_slope=0.2)
        self.fc2 = layers.Dense(output_dim, activation=None, kernel_initializer='he_uniform')

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
        x = self.activation(x)
        x = self.fc2(x)

        x += residual
        if final_nl:
            x = self.activation(x)
        return x


class ImprovedActor(tf.keras.Model):
    def __init__(self, state_dim, action_dim, n_neurons=512):
        super(ImprovedActor, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.rb1 = ResBlock(state_dim, state_dim, n_neurons)
        self.rb2 = ResBlock(state_dim + state_dim, state_dim + state_dim, n_neurons)

        self.action_low  = tf.constant([0.05, -0.82], dtype=tf.float32)
        self.action_high = tf.constant([0.26,  0.82], dtype=tf.float32)
        self.dropout = layers.Dropout(rate=0.1)
        self.mu_layer = layers.Dense(action_dim, activation='tanh')  # Ограничим действия в диапазоне [-1, 1]
        self.log_std_layer = layers.Dense(
            action_dim,
            activation='softplus',
            kernel_initializer='he_uniform',
            bias_initializer=initializers.Constant(-1.0)
        )

    def call(self, obs, training=False, raw_actions = None):
        if isinstance(obs, np.ndarray):
            obs = tf.convert_to_tensor(obs, dtype=tf.float32)

        if len(obs.shape) == 1:
            obs = tf.expand_dims(obs, axis=0)

        x0 = obs
        x = self.rb1(x0)
        x = self.rb2(tf.concat([x0, x], axis=-1))
        x = self.dropout(x, training=training)

        mu = self.mu_layer(x)
        mu = tf.clip_by_value(mu, -1.0, 1.0)

        log_std = self.log_std_layer(x)
        std = tf.exp(log_std)
        std = tf.clip_by_value(std, 1e-6, 1.5) 

        base_dist = tfp.distributions.Normal(loc=mu, scale=std)
        dist = tfp.distributions.TransformedDistribution(
            distribution=base_dist,
            bijector=tfp.bijectors.Tanh()
        )
        if raw_actions is None:


            raw_action = dist.sample()
            log_prob = tf.reduce_sum(dist.log_prob(raw_action), axis=-1)

            raw_action_clipped = tf.clip_by_value(raw_action, -0.999, 0.999)
            action_scaled = self.action_low + (raw_action_clipped + 1.0) * 0.5 * (self.action_high - self.action_low)
           
            entropy  = tf.reduce_sum(base_dist.entropy(), axis=-1)

            return action_scaled, log_prob, entropy, std, raw_action
        else:

            log_prob = tf.reduce_sum(dist.log_prob(raw_actions), axis=-1)
            entropy  = tf.reduce_sum(base_dist.entropy(), axis=-1)

            return log_prob, entropy
    
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
            action_dim=config.get("action_dim", 2),  
            n_neurons=config.get("n_neurons", 512),
            # dtype=config.get("dtype", "float32")
        )