import gym
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
import rclpy
from env import TurtleBotEnv
from actor import ImprovedActor
from critic import ImprovedCritic, StaticCritic, world_to_map
from config import TARGET_X, TARGET_Y
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt
import logging


logging.basicConfig(filename='training_logs.txt', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def precompute_value_map(grid_map, optimal_path, goal, path_weight=5.0, obstacle_weight=1.0, goal_weight=4.0):
    """ Предобученная карта значений для статичного критика """
    height, width = grid_map.shape
    value_map = np.zeros((height, width))

    # Препятствия
    obstacle_mask = (grid_map == 1)
    obstacle_distances = distance_transform_edt(obstacle_mask)
    obstacle_distances = np.clip(obstacle_distances, 1e-3, None)

    # Расстояние до пути
    path_mask = np.zeros_like(grid_map, dtype=bool)
    for x, y in optimal_path:
        path_mask[y, x] = True
    path_distances = distance_transform_edt(~path_mask)
    max_path = np.max(path_distances)
    path_values = max_path - path_distances  # ближе — больше
    path_values = (path_values - path_values.min()) / (path_values.max() - path_values.min() + 1e-8)

    # Расстояние до цели
    goal_x, goal_y = world_to_map(goal, resolution=0.05, origin=(-4.86, -7.36),
                                   map_offset=(45, 15), map_shape=grid_map.shape)
    goal_mask = np.zeros_like(grid_map, dtype=bool)
    goal_mask[goal_y, goal_x] = True
    goal_distances = distance_transform_edt(~goal_mask)
    max_goal = np.max(goal_distances)
    goal_values = max_goal - goal_distances
    goal_values = (goal_values - goal_values.min()) / (goal_values.max() - goal_values.min() + 1e-8)

    # Обратная зависимость от близости к препятствиям
    repulsion_values = np.exp(-obstacle_distances / 1.5)
    repulsion_values = (repulsion_values - repulsion_values.min()) / (repulsion_values.max() - repulsion_values.min() + 1e-8)
    # Комбинируем
    value_map = (
        path_weight * path_values +
        goal_weight * goal_values -
        obstacle_weight * repulsion_values
    )

    # Нормализуем итоговую карту значений (опционально)
    value_map = (value_map - value_map.min()) / (value_map.max() - value_map.min() + 1e-8)

    return value_map

def plot_value_map(value_map):
        # plt.figure(figsize=(8, 6))
        # plt.imshow(value_map, cmap="viridis")
        # plt.colorbar(label="Value")
        # plt.title("Critic Value Map")
        # plt.show()
        pass


# --- Класс агента PPO ---
class PPOAgent:
    def __init__(self, env):
        self.env = env
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        self.optimal_path = env.optimal_path
        self.grid_map = env.grid_map

        self.goal = np.array(env.goal, dtype=np.float32)
        # print(self.goal)
 
        self.x_range = np.array(env.x_range, dtype=np.float32)  # Диапазон X как массив NumPy
        self.y_range = np.array(env.y_range, dtype=np.float32)  # Диапазон Y как массив NumPy
        

        # self.obstacles = np.array(env.obstacles, dtype=np.float32)
        # print(self.obstacles)

        # Коэффициенты
        self.gamma = 0.995  # коэффициент дисконтирования
        self.epsilon = 0.15 # параметр клиппинга
        self.actor_lr = 0.0003
        self.critic_lr = 0.0003
        self.gaelam = 0.95
        self.min_alpha = 0.1
        self.max_alpha = 0.9 
        # self.alpha = 0.1

        # Модели
        self.actor = ImprovedActor(self.state_dim, self.action_dim)
        self.critic = ImprovedCritic(self.state_dim, grid_map=self.grid_map, optimal_path=self.optimal_path)
        self.value_map = precompute_value_map(self.grid_map, self.optimal_path, self.goal)
        self.critic_st = StaticCritic(self.value_map, self.grid_map) 

        # self.value_map = self.critic.initialize_value_map(grid_map=self.grid_map)  
        
        # print(type(self.value_map))  # Должно быть <class 'numpy.ndarray'>
        # print(self.value_map.dtype)  # Должно быть float32 или float64
        # print(self.value_map.shape)

        # print(self.value_map)

        plot_value_map(self.value_map)

        # Оптимизаторы
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=self.actor_lr)
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=self.critic_lr)
    
    def update_alpha(self, episode, max_episodes):
        alpha = self.max_alpha - (self.max_alpha - self.min_alpha) * (episode / max_episodes)
        alpha = np.clip(alpha, self.min_alpha, self.max_alpha)
        return np.float32(alpha)
    
    def get_action(self, state, alpha, epsilon):
        state_tensor = tf.convert_to_tensor(state.reshape(1, -1), dtype=tf.float32)
        action_raw, log_prob, entropy, _, _ = self.actor(state_tensor)
        action = action_raw.numpy().reshape(-1)

        return action, log_prob.numpy().squeeze(), entropy.numpy().squeeze()
    

    # Вычисление преимущесвт и возврата
    def compute_advantages(self, rewards, values, dones):
        # print('Rewrds: ', rewards)
        # print('Values:', values) angle_diff
        # print('Next values:', next_value)
        advantages = np.zeros_like(rewards)
        returns = np.zeros_like(rewards)
        last_gae = 0
        next_value = values[-1]
        for t in reversed(range(len(rewards))):
            # Обработка последнего шага
            if t == len(rewards) - 1:
                next_value = values[-1]
                next_done = dones[t]  
            else:
                # Обработка остальных шагов
                next_value = values[t + 1]
                next_done = dones[t + 1]

            # Вычисление ошибки предсказания
            delta = rewards[t] + self.gamma * next_value * (1 - next_done) - values[t]
            # if np.isnan(delta):
            #     print(f"NaN in delta: rewards[{t}]={rewards[t]}, next_value={next_value}, values[{t}]={values[t]}")
            advantages[t] = last_gae = delta + self.gamma * self.gaelam * (1 - next_done) * last_gae
            returns[t] = advantages [t] + values[t]  
        # print('Advanteges:', advantages)
    # Возвраты для обновления критика
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-10)
        # returns = advantages + values[:-1]  
        # print ('Returns:', returns)
        logger.info(f'Returns:  {returns}' )
        logger.info(f'Advanteges:  {advantages}' )
        logger.info(f'Values_learned: {values}')

        return advantages, returns 
    
    # Обновление политик
    def update(self, states, actions, advantages, returns, log_probs_old, values_static, alpha):
        states = tf.convert_to_tensor(states, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.float32)
        log_probs_old = tf.convert_to_tensor(log_probs_old, dtype=tf.float32)
        advantages = tf.convert_to_tensor(advantages, dtype=tf.float32)
        returns = tf.convert_to_tensor(returns, dtype=tf.float32)
        values_static = tf.convert_to_tensor(values_static, dtype=tf.float32)

        entropy_coef = 0.01  # по желанию: можно сделать адаптивным

        # === Actor update ===
        with tf.GradientTape() as tape:
            _, log_probs, entropy, _, _ = self.actor(states)
            log_probs = tf.reduce_sum(log_probs, axis=-1)
            ratios = tf.exp(log_probs - log_probs_old)
            clipped_ratios = tf.clip_by_value(ratios, 1.0 - self.epsilon, 1.0 + self.epsilon)
            surrogate1 = ratios * advantages
            surrogate2 = clipped_ratios * advantages
            actor_loss = -tf.reduce_mean(tf.minimum(surrogate1, surrogate2))

            # ➕ добавим энтропийный бонус
            entropy_bonus = tf.reduce_mean(entropy)
            actor_loss -= entropy_coef * entropy_bonus

        actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))

        # === Critic update ===
        with tf.GradientTape() as tape:
            
            values = self.critic.call(states)  # shape (batch, 1)
            # values = tf.squeeze(values, axis=1)  # shape (batch,)
            # values_static = tf.broadcast_to(values_static, values.shape)
            critic_loss = tf.reduce_mean(tf.square(returns - values))
            hint_loss = tf.reduce_mean(tf.square(values_static - values))
            total_critic_loss = critic_loss + alpha * hint_loss

        critic_grads = tape.gradient(total_critic_loss, self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))

    def train(self, max_episodes=500, batch_size=32):
        all_rewards = []
        alpha = 0
        epsilon_min = 0.01
        epsilon_decay = 0.99
        epsilon = 0.2
        for episode in range(max_episodes):
            state = np.reshape(self.env.reset(), [1, self.state_dim])
            episode_reward = 0
            done = False

            states, actions, rewards, dones, probs = [], [], [], [], []
            values_learned, values_static = [], []

            while not done:
                action, log_prob, _ = self.get_action(state, alpha, epsilon)
                logger.info(f'Action:  {action}')
                logger.info(f'Prob:  {log_prob}')
                next_state, reward, done, _ = self.env.step(action)
                
                if np.isnan(next_state).any():
                    print("Обнаружен NaN в состоянии!")
                    break
                
                next_state = np.reshape(next_state, [1, self.state_dim])

                value_learned = self.critic.call(state)[0, 0]
                value_static = self.critic_st.call(state)  # StaticCritic
                
                logger.info(f'Value learned:  {value_learned}')
                logger.info(f'Value static:  {value_static}')

                states.append(state)
                actions.append(action)
                rewards.append(reward)
                dones.append(done)
                probs.append(log_prob)
                values_learned.append(value_learned)
                values_static.append(value_static)

                state = next_state
                episode_reward += reward
                logger.info(f'Episode reward  {episode_reward}')

            epsilon = max(epsilon_min, epsilon * epsilon_decay)
            if len(rewards) < 10:
                continue

            # Последнее значение критика
            next_value_learned = self.critic.call(next_state)[0, 0]
            next_value_static = self.critic_st.call(next_state)

            values_learned.append(next_value_learned)
            values_static.append(next_value_static)

            # Комбинируем значения критиков
            # values_combined = self.alpha * np.array(values_learned) + (1 - self.alpha) * np.array(values_static)

            # Вычисляем `advantages` и `returns`
            # print(alpha)
            advantages, returns = self.compute_advantages(rewards, values_learned, dones)
            alpha = self.update_alpha(episode, max_episodes)
            # Обновляем модель
            states = np.vstack(states).reshape(-1, self.state_dim)
            self.update(np.vstack(states), actions, advantages, returns, probs, values_static, alpha)

            all_rewards.append(episode_reward)
            print(f'Episode {episode + 1}, Reward: {episode_reward}')

        # dummy_input = np.zeros((1, self.state_dim))  # Подставь размер входа
        # self.actor(dummy_input)  # Прогоняем модель с фиктивными данными
        self.actor.save('ppo_turtlebot_actor.keras')
        # self.critic.save('ppo_turtlebot_critic.keras')

def main(args=None):
    rclpy.init(args=args)
    env = TurtleBotEnv()
    agent = PPOAgent(env)
    agent.train()
    rclpy.shutdown()

if __name__ == '__main__':
    main()