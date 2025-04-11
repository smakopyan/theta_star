import gym
import numpy as np
import tensorflow as tf
# from tensorflow import kears
from keras import layers
import rclpy
from .turtlebot_env import TurtleBotEnv
from .critic import ImprovedCritic, StaticCritic, world_to_map
from .actor import ImprovedActor
##from my_turtlebot_package.config import TARGET_X, TARGET_Y
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt
import logging


logging.basicConfig(filename='training_logs.txt', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def precompute_value_map(grid_map, optimal_path, goal, path_weight = 5.0, obstacle_weight=10.0, goal_weight = 4.0):
        """ Заполняем таблицу значений критика для всех точек grid_map """
        height, width = grid_map.shape
        value_map = np.zeros((height, width))

        # Вычисляем расстояние от каждой точки карты до ближайшего препятствия
        obstacle_mask = (grid_map == 1)
        obstacle_distances = distance_transform_edt(obstacle_mask)  # Чем ближе к препятствию, тем меньше значение

        # Вычисляем расстояние от каждой точки до ближайшей точки пути
        path_mask = np.zeros_like(grid_map, dtype=bool)
        for x, y in optimal_path:  # optimal_path — список (x, y)
            path_mask[y, x] = True
        path_distances = distance_transform_edt(~path_mask)  # Чем ближе к пути, тем меньше значение

        goal_x, goal_y = world_to_map(goal, resolution=0.05, origin=(-4.86, -7.36),
                                    map_offset=(45, 15), map_shape= grid_map.shape)
        # print(goal_x, goal_y)
        goal_mask = np.zeros_like(grid_map, dtype=bool)
        goal_mask[goal_y, goal_x] = True
        goal_distances = distance_transform_edt(~goal_mask)  # Чем ближе к цели, тем меньше значение

        # obstacle_distances = np.clip(obstacle_distances, 0, 5)  # Обрезаем максимальные значения
        # path_distances = np.clip(path_distances, 0, 5)
        # goal_distances = np.clip(goal_distances, 0, 5)

        # Создаём градиентное поле
        value_map = -path_distances * path_weight - obstacle_distances * (obstacle_weight / (obstacle_distances + 1)) - goal_distances * goal_weight

        return value_map

def plot_value_map(value_map):
        plt.figure(figsize=(8, 6))
        plt.imshow(value_map, cmap="viridis")
        plt.colorbar(label="Value")
        plt.title("Critic Value Map")
        plt.show()

# --- Класс агента PPO ---
class PPOAgent:
    def __init__(self, env):
        self.env = env
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n
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
    
    def update_alpha(self,episode, max_episodes):
        alpha = self.min_alpha + (self.max_alpha - self.min_alpha) * (episode / max_episodes)
        alpha = np.clip(alpha, self.min_alpha, self.max_alpha)
        alpha = np.float32(alpha)
        return alpha
    
    def get_action(self, state, value_map, alpha, epsilon):
        state = np.reshape(state, [1, self.state_dim])
        logger.info(f'State in get_action: {state}') 
        prob = self.actor(state).numpy().squeeze()
        prob = np.nan_to_num(prob, nan=1.0/self.action_dim)
        prob /= np.sum(prob)  # Нормализация

        action_values = np.zeros(self.action_dim)

        for action in range(self.action_dim):
            next_state = self.env.get_next_state(state, action, self.env.current_yaw)

            # Оценка текущего состояния
            value_current_learned = self.critic.call(state)
            value_current_static = self.critic_st.call(state)

            # Оценка будущего состояния
            value_next_learned = self.critic.call(next_state)
            value_next_static = self.critic_st.call(next_state)

            # Взвешенное объединение критиков
            value_current = alpha * value_current_learned + (1 - alpha) * value_current_static
            value_next = alpha * value_next_learned + (1 - alpha) * value_next_static

            # Усреднение текущей и будущей оценки
            action_values[action] = 0.3 * value_current + 0.7 * value_next
            logger.info(f'Action values in get_action before normalization: {action_values}') 

        # Нормализация оценок критика
        action_values = (action_values - np.min(action_values)) / (np.max(action_values) - np.min(action_values) + 1e-10)
        logger.info(f'Action values in get_action after normalization: {action_values}')  

        combined_scores = alpha * prob + (1 - alpha) * action_values
        logger.info(f'Combined scores in get_action: {combined_scores}') 

            # Эпсилон-жадный выбор действия
        # if np.random.rand() < self.epsilon:
        #     action = np.random.choice(self.action_dim)
        # else:
        #     action = np.argmax(combined_scores)
        action = np.argmax(combined_scores)

        return action, prob
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
    def update(self, states, actions, advantages, returns, old_probs, values_static):
        states = tf.convert_to_tensor(states, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.int32)
        advantages = tf.convert_to_tensor(advantages, dtype=tf.float32)
        returns = tf.convert_to_tensor(returns, dtype=tf.float32)
        old_probs = tf.convert_to_tensor(old_probs, dtype=tf.float32)

        old_probs = tf.reduce_sum(old_probs * tf.one_hot(actions, depth=self.action_dim), axis=1)

        # === Обновление актора ===
        with tf.GradientTape() as tape:
            prob = self.actor(states)
            chosen_probs = tf.reduce_sum(prob * tf.one_hot(actions, depth=self.action_dim), axis=1)

            prob_ratio = chosen_probs / old_probs
            clipped_prob_ratio = tf.clip_by_value(prob_ratio, 1.0 - self.epsilon, 1.0 + self.epsilon)

            surrogate_loss = tf.minimum(prob_ratio * advantages, clipped_prob_ratio * advantages)
            logger.info(f'Surrogate loss: {surrogate_loss}')
            actor_loss = -tf.reduce_mean(surrogate_loss)
            logger.info(f'Acotor loss: {actor_loss}')
            
        actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))

        # === Обновление критика ===
        with tf.GradientTape() as tape:
            # Получаем значения из критика
            values = self.critic.call(states)
            # print(values)
            # Рассчитываем потерю критика
            critic_loss = tf.reduce_mean(tf.square(returns - values))
            hint_loss = tf.reduce_mean(tf.square(values_static - values))
            # Итоговый лосс критика
            total_critic_loss = critic_loss + 0.1 * hint_loss


        # print(self.critic.trainable_variables)
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
                alpha = self.update_alpha(episode, max_episodes)
                action, prob = self.get_action(state, self.value_map, alpha, epsilon)
                logger.info(f'Action:  {action}')
                logger.info(f'Prob:  {prob}')
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
                probs.append(prob)
                values_learned.append(value_learned)
                values_static.append(value_static)

                state = next_state
                episode_reward += reward
                logger.info(f'Episode reward  {episode_reward}')

            epsilon = max(epsilon_min, epsilon * epsilon_decay)

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

            # Обновляем модель
            self.update(np.vstack(states), actions, advantages, returns, probs, values_static)

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