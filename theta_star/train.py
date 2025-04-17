import numpy as np
import tensorflow as tf
from keras import layers
import rclpy
from turtlebot_env import TurtleBotEnv
from critic import ImprovedCritic, StaticCritic, world_to_map
from actor import ImprovedActor
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt
import logging
import signal
import sys
import traceback
from ament_index_python.packages import get_package_share_directory
import os


training_logger = logging.getLogger('training')
training_logger.setLevel(logging.INFO)
training_handler = logging.FileHandler('training_logs.txt')
training_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
training_handler.setFormatter(training_formatter)
training_logger.addHandler(training_handler)

reward_logger = logging.getLogger('reward')
reward_logger.setLevel(logging.INFO)
reward_handler = logging.FileHandler('reward_logs.txt')
reward_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
reward_handler.setFormatter(reward_formatter)
reward_logger.addHandler(reward_handler)

training_logger.info("Сообщение о процессе обучения")
reward_logger.info("Сообщение о наградах")

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

        goal_x, goal_y = world_to_map(goal, resolution=0.05, origin=(-7.76, -7.15),
                                    map_offset=(0, 0), map_shape= grid_map.shape)
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
    pass
        # plt.figure(figsize=(8, 6))
        # plt.imshow(value_map, cmap="viridis")
        # plt.colorbar(label="Value")
        # plt.title("Critic Value Map")
        # plt.show()

# --- Класс агента PPO ---
class PPOAgent:
    def __init__(self, env):
        self.env = env
        self.num_robots = env.num_robots

        self.state_dims = [env.observation_space.shape[0], env.observation_space.shape[0]]

        self.action_dims = [env.action_space.n, env.action_space.n]
        # self.optimal_path = env.optimal_path
        self.grid_map = env.grid_map

        # self.goal = np.array(env.goal, dtype=np.float32)
        # print(self.goal)
 
        self.x_range = np.array(env.x_range, dtype=np.float32)  # Диапазон X как массив NumPy
        self.y_range = np.array(env.y_range, dtype=np.float32)  # Диапазон Y как массив NumPy
        

        # self.obstacles = np.array(env.obstacles, dtype=np.float32)
        # print(self.obstacles)

        # Коэффициенты
        self.gamma = 0.995  # коэффициент дисконтирования
        self.epsilon = 0.15 # параметр клиппинга
        self.actor_lrs = [0.0003] * self.num_robots
        self.critic_lrs = [0.0003] * self.num_robots
        self.gaelam = 0.95
        self.min_alpha = 0.1
        self.max_alpha = 0.9 
        # self.alpha = 0.1

        # for state_dim, action_dim in zip(self.state_dims, self.action_dims):
        #     print(state_dim, action_dim)

        self.actors = [
            ImprovedActor(state_dim, action_dim) 
            for state_dim, action_dim in zip(self.state_dims, self.action_dims)
        ]

            

        self.critics = [
            ImprovedCritic(state_dim, env.grid_map, env.robots[i].optimal_path)
            for i, state_dim in enumerate(self.state_dims)
        ]
 

        self.value_maps = [
            precompute_value_map(env.grid_map, robot.optimal_path, robot.goal)
            for robot in env.robots
        ]        
        self.critics_st = [StaticCritic(value_map, self.grid_map) 
                           for value_map in self.value_maps]

        # self.value_map = self.critic.initialize_value_map(grid_map=self.grid_map)  
        
        # print(type(self.value_map))  # Должно быть <class 'numpy.ndarray'>
        # print(self.value_map.dtype)  # Должно быть float32 или float64
        # print(self.value_map.shape)

        # print(self.value_map)
        for value_map in self.value_maps:
            plot_value_map(value_map)

        # Оптимизаторы
        self.actor_optimizers = [
            tf.keras.optimizers.Adam(learning_rate=lr) 
            for lr in self.actor_lrs
        ]
        
        self.critic_optimizers = [
            tf.keras.optimizers.Adam(learning_rate=lr)
            for lr in self.critic_lrs
        ]

    def update_alpha(self,episode, max_episodes):
        alpha = self.min_alpha + (self.max_alpha - self.min_alpha) * (episode / max_episodes)
        alpha = np.clip(alpha, self.min_alpha, self.max_alpha)
        alpha = np.float32(alpha)
        return alpha
    
    def get_action(self, states, alpha, epsilon):
        actions = []
        probs = []
        for i in range(self.num_robots):
            state = states[i]
            robot = self.env.robots[i]
            state_tensor = tf.convert_to_tensor([state], dtype=tf.float32)

            training_logger.info(f'State in get_action: {state}') 
            prob = self.actors[i](state_tensor).numpy().squeeze()
            prob = np.nan_to_num(prob, nan=1.0/self.action_dims[i])
            prob /= np.sum(prob)

            action_values = np.zeros(self.action_dims[i])

            for action in range(self.action_dims[i]):
                next_state = self.env.get_next_state(
                    state, action, self.env.robots[i].current_yaw
                )
                # Оценка текущего состояния
                value_current_learned = self.critics[i].call(state)
                value_current_static = self.critics_st[i].call(state)

                # Оценка будущего состояния
                value_next_learned = self.critics[i].call(next_state)
                value_next_static = self.critics_st[i].call(next_state)

                # Взвешенное объединение критиков
                value_current = alpha * value_current_learned + (1 - alpha) * value_current_static
                value_next = alpha * value_next_learned + (1 - alpha) * value_next_static

                # Усреднение текущей и будущей оценки
                action_values[action] = 0.3 * value_current + 0.7 * value_next
                training_logger.info(f'Action values in get_action before normalization: {action_values}') 

            # Нормализация оценок критика
            action_values = (action_values - np.min(action_values)) / (np.max(action_values) - np.min(action_values) + 1e-10)
            training_logger.info(f'Action values in get_action after normalization: {action_values}')  

            combined_scores = alpha * prob + (1 - alpha) * action_values
            training_logger.info(f'Combined scores in get_action: {combined_scores}') 

                # Эпсилон-жадный выбор действия
            # if np.random.rand() < self.epsilon:
            #     action = np.random.choice(self.action_dim)
            # else:
            #     action = np.argmax(combined_scores)
            action = np.argmax(combined_scores)
            actions.append(action)
            probs.append(prob)

        return actions, probs
    
    # Вычисление преимущесвт и возврата
    def compute_advantages(self, rewards, values, dones):
        # print('Rewrds: ', rewards)
        # print('Values:', values) angle_diff
        # print('Next values:', next_value)
        advantages = []
        returns = []
        for i in range(self.num_robots):
            robot_advantages = np.zeros_like(rewards[i])
            robot_returns = np.zeros_like(rewards[i])
            last_gae = 0
            next_value = values[i][-1]
            
            for t in reversed(range(len(rewards[i]))):
                # Обработка последнего шага
                if t == len(rewards[i]) - 1:
                    next_value = values[i][-1]
                    next_done = dones[i][t]  
                else:
                    # Обработка остальных шагов
                    next_value = values[i][t + 1]
                    next_done = dones[i][t + 1]

                # Вычисление ошибки предсказания
                delta = rewards[i][t] + self.gamma * next_value * (1 - next_done) - values[i][t]
                # if np.isnan(delta):
                #     print(f"NaN in delta: rewards[{t}]={rewards[t]}, next_value={next_value}, values[{t}]={values[t]}")
                robot_advantages[t] = last_gae = delta + self.gamma * self.gaelam * (1 - next_done) * last_gae
                robot_returns[t] = robot_advantages [t] + values[i][t]  
            # print('Advanteges:', advantages)
        # Возвраты для обновления критика
            robot_advantages = (robot_advantages - np.mean(robot_advantages)) / (np.std(robot_advantages) + 1e-10)
            # returns = advantages + values[:-1]  
            # print ('Returns:', returns)
            advantages.append(robot_advantages)
            returns.append(robot_returns)
            training_logger.info(f'Returns:  {returns}' )
            training_logger.info(f'Advanteges:  {advantages}' )
            training_logger.info(f'Values_learned: {values}')

        return advantages, returns 
    
    # Обновление политик
    def update(self, batch_data):
        for i in range(self.num_robots):
            states = tf.convert_to_tensor(batch_data[i]['states'], dtype=tf.float32)
            actions = tf.convert_to_tensor(batch_data[i]['actions'], dtype=tf.int32)
            advantages = tf.convert_to_tensor(batch_data[i]['advantages'], dtype=tf.float32)
            returns = tf.convert_to_tensor(batch_data[i]['returns'], dtype=tf.float32)
            old_probs = tf.convert_to_tensor(batch_data[i]['probs'], dtype=tf.float32)
            old_probs = tf.reduce_sum(old_probs * tf.one_hot(actions, depth=self.action_dims[0]), axis=1)

            # === Обновление актора ===
            with tf.GradientTape() as tape:
                prob = self.actors[i](states)
                chosen_probs = tf.reduce_sum(prob * tf.one_hot(actions, depth=self.action_dims[0]), axis=1)

                prob_ratio = chosen_probs / old_probs
                clipped_prob_ratio = tf.clip_by_value(prob_ratio, 1.0 - self.epsilon, 1.0 + self.epsilon)

                surrogate_loss = tf.minimum(prob_ratio * advantages, clipped_prob_ratio * advantages)
                training_logger.info(f'Surrogate loss: {surrogate_loss}')
                actor_loss = -tf.reduce_mean(surrogate_loss)
                training_logger.info(f'Acotor loss: {actor_loss}')
                
            actor_grads = tape.gradient(actor_loss, self.actors[i].trainable_variables)
            self.actor_optimizers[i].apply_gradients(zip(actor_grads, self.actors[i].trainable_variables))

            # === Обновление критика ===
            with tf.GradientTape() as tape:
                # Получаем значения из критика
                values = self.critics[i].call(states)
                values_static = batch_data[i]['values_static']
                # print(values)
                # Рассчитываем потерю критика
                critic_loss = tf.reduce_mean(tf.square(returns - values))
                hint_loss = tf.reduce_mean(tf.square(values_static - values))
                # Итоговый лосс критика
                total_critic_loss = critic_loss + 0.1 * hint_loss


            # print(self.critic.trainable_variables)
            critic_grads = tape.gradient(total_critic_loss, self.critics[i].trainable_variables)
            self.critic_optimizers[i].apply_gradients(
                zip(critic_grads, self.critics[i].trainable_variables)
            )
    def save_models(self):
        for i in range(self.num_robots):
            self.actors[i].save(f'ppo_actor_robot_{i}.keras')
            self.critics[i].save(f'ppo_critic_robot_{i}.keras')
        print("\nModels saved successfully!")

    def train(self, max_episodes=500, batch_size=32, num_robots=2):
        all_rewards = []
        alpha = 0
        epsilon_min = 0.01
        epsilon_decay = 0.99
        epsilon = 0.2
        for episode in range(max_episodes):
            print('----------------------------------------------------------------------------')
            print("Episode: ",episode)
            training_logger.info('\n----------------------------------------------------------------------------')
            reward_logger.info('\n----------------------------------------------------------------------------')
            training_logger.info(f'Episode {episode}')
            reward_logger.info(f'Episode {episode}')

            states = self.env.reset()
            episode_rewards = [0] * num_robots
            done = False

            batch_data = {i: {
                'states': [], 
                'actions': [], 
                'rewards': [],
                'dones': [], 
                'values_learned': [], 
                'values_static': [], 
                'probs': []
            } for i in range(num_robots)}
            
            step = 0

            while not done:
                alpha = self.update_alpha(episode, max_episodes)
                actions, probs = self.get_action(states, alpha, epsilon)
                # logger.info(f'Action:  {action}')
                # logger.info(f'Prob:  {prob}')
                next_states, rewards, dones, _ = self.env.step(actions)
                if isinstance(dones, bool):
                    dones = [dones] * self.num_robots
                if np.isnan(next_states).any():
                    print("Обнаружен NaN в состоянии!")
                    break
                
                for i in range(self.num_robots):
                    state_tensor = tf.convert_to_tensor([states[i]], dtype=tf.float32)
                    value_learned = self.critics[i](state_tensor).numpy()[0][0]
                    value_static = self.critics_st[i].call(states[i])

                    batch_data[i]['states'].append(states[i])
                    batch_data[i]['actions'].append(actions[i])
                    batch_data[i]['rewards'].append(rewards[i])
                    batch_data[i]['dones'].append(dones[i])
                    batch_data[i]['values_learned'].append(value_learned)
                    batch_data[i]['values_static'].append(value_static)
                    batch_data[i]['probs'].append(probs[i])

                    episode_rewards[i] += rewards[i]
                    training_logger.info(f'Episode reward  {episode_rewards[i]}')
                states = next_states
                step += 1

                done = any(dones) or (step >= self.env.max_steps)
                if done or step % batch_size == 0:
                    advantages, returns = self.compute_advantages(
                        [batch_data[i]['rewards'] for i in range(self.num_robots)],
                        [batch_data[i]['values_learned'] for i in range(self.num_robots)],
                        [batch_data[i]['dones'] for i in range(self.num_robots)]
                    )

                    for i in range(self.num_robots):
                        batch_data[i]['advantages'] = advantages[i]
                        batch_data[i]['returns'] = returns[i]

                    self.update(batch_data)

                    for i in range(self.num_robots):
                        batch_data[i] = {k: [] for k in batch_data[i]}

                    if done:
                        break

            epsilon = max(epsilon_min, epsilon * epsilon_decay)

            all_rewards.append(episode_rewards)
            avg_reward = np.mean(episode_rewards)
            print(f"Episode {episode+1}/{max_episodes}")
            print(f"  Robot 1 Reward: {episode_rewards[0]:.2f}")
            print(f"  Robot 2 Reward: {episode_rewards[1]:.2f}")
            print(f"  Average Reward: {avg_reward:.2f}")
            reward_logger.info(f"Episode {episode+1}/{max_episodes}")
            reward_logger.info(f"  Robot 1 Reward: {episode_rewards[0]:.2f}")
            reward_logger.info(f"  Robot 2 Reward: {episode_rewards[1]:.2f}")
            reward_logger.info(f"  Average Reward: {avg_reward:.2f}")
            reward_logger.info("="*50)

        for i in range(self.num_robots):
            self.actors[i].save(f'ppo_actor_robot_{i}.keras')
            self.critics[i].save(f'ppo_critic_robot_{i}.keras')

        return all_rewards

def main(args=None):
    rclpy.init(args=args)
    env = TurtleBotEnv()
    agent = PPOAgent(env)
    
    def signal_handler(sig, frame):
        print("\nCtrl+C detected. Saving models...")
        agent.save_models()
        rclpy.shutdown()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        agent.train()
    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving models...")
        agent.save_models()
    except Exception as e:  
        print(f"\nCritical error occurred: {str(e)}", file=sys.stderr)
        agent.save_models()

        traceback.print_exc()  # Печать трейса ошибки
    finally: 
        rclpy.shutdown()
if __name__ == '__main__':
    main()