import numpy as np
import tensorflow as tf
from keras import layers
import rclpy
from turtlebot_env import TurtleBotEnv, Zfilter, RunningStat
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
import signal
import subprocess
import sys
from tensorflow import summary
from datetime import datetime

# --- Класс агента PPO ---
class PPOAgent:
    def __init__(self, env):
        self.env = env
        self.num_robots = env.num_robots
        self.state_dim = env.observation_space.shape[0]
        self.cent_state_dim = self.state_dim * self.num_robots 
        self.state_dims = [env.observation_space.shape[0], env.observation_space.shape[0]]

        self.action_dims = [env.action_space.shape[0], env.action_space.shape[0]]
        self.grid_map = env.grid_map

        self.x_range = np.array(env.x_range, dtype=np.float32)  # Диапазон X как массив NumPy
        self.y_range = np.array(env.y_range, dtype=np.float32)  # Диапазон Y как массив NumPy
        
        self.base_state_dim = env.observation_space.shape[0]

        # Коэффициенты
        self.gamma = 0.99  # коэффициент дисконтирования
        self.epsilon = 0.15 # параметр клиппинга
        self.actor_lrs = [0.0003] * self.num_robots
        self.critic_lr = 0.0003
        self.gaelam = 0.95
        self.min_entropy = 0.0001
        self.max_entropy = 0.01

        self.best_models = [0, 0]
        self.actors = [
            ImprovedActor(state_dim, action_dim) 
            for state_dim, action_dim in zip(self.state_dims, self.action_dims)
        ]

        self.critic = ImprovedCritic(self.cent_state_dim, env.grid_map, env.robots[0].optimal_path)

        self.actor_optimizers = [
            tf.keras.optimizers.Adam(learning_rate=lr) 
            for lr in self.actor_lrs
        ]
        
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=self.critic_lr)

        current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.log_dir = f'logs/ppo/{current_time}'
        self.summary_writers = {
            i: summary.create_file_writer(f'{self.log_dir}/robot_{i}')
            for i in range(self.num_robots)
        }

        self.global_step = 0
        self.state_filter = Zfilter(prev_filter=None, shape=self.env.observation_space.shape[0], clip=1.0)

    def get_action(self, states):
        actions = []
        log_probs = []
        entropys = []
        stds = []
        r_actions = []
        for i in range(self.num_robots):
            state = states[i]
            state_tensor = tf.convert_to_tensor(state.reshape(1, -1), dtype=tf.float32)
            action_raw, log_prob, entropy, std, raw_action = self.actors[i](state_tensor, raw_actions = None)
            action = action_raw.numpy().squeeze()
            r_action = raw_action.numpy().squeeze()
            actions.append(action)
            log_probs.append(log_prob.numpy().squeeze())
            entropys.append(entropy.numpy().squeeze())
            stds.append(std)
            r_actions.append(r_action)
        return actions, log_probs, entropys, stds, r_actions   

    

    def compute_advantages(self, all_rewards, values, all_dones):
        last_gae = 0
        next_value = values[-1]
        rewards = []
        dones = []
        for i, j in zip(all_rewards[0], all_rewards[1]):
            rewards.append(i+j)
        for i, j in zip(all_dones[0], all_dones[1]):
            dones.append(any([i, j]))
        returns = np.zeros(len(rewards))
        advantages = np.zeros(len(rewards))

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = values[-1]
                next_done = dones[t]  
            else:
                next_value = values[t + 1]
                next_done = dones[t + 1]

            delta = rewards[t] + self.gamma * next_value * (1 - next_done) - values[t]
            # if np.isnan(delta):
            #     print(f"NaN in delta: rewards[{t}]={rewards[t]}, next_value={next_value}, values[{t}]={values[t]}")
            advantages[t] = last_gae = delta + self.gamma * self.gaelam * (1 - next_done) * last_gae
            returns[t] = advantages[t] + values[t]  
        # print('Advanteges:', advantages)
    # Возвраты для обновления критика
        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-10)
        # returns = advantages + values[:-1]  
        # print ('Returns:', returns)
        return advantages, returns 
    # Обновление политик
    def update(self, states, actions, advantages, returns, log_probs_old, entropy_coef, raw_actions, agent_ind):
        states = tf.convert_to_tensor(states, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.float32)
        log_probs_old = tf.convert_to_tensor(log_probs_old, dtype=tf.float32)
        advantages = tf.convert_to_tensor(advantages, dtype=tf.float32)
        raw_actions = tf.convert_to_tensor(raw_actions, dtype=tf.float32)

        # === Actor update ===
        with tf.GradientTape() as tape:
            log_probs, entropy = self.actors[agent_ind](states, training = True, raw_actions = raw_actions)
            log_probs = tf.debugging.check_numerics(log_probs, message="log_probs contain NaN")
            # log_probs = tf.reduce_sum(log_probs, axis=-1)
            ratios = tf.exp(log_probs - log_probs_old)
            ratios = tf.debugging.check_numerics(ratios, message="ratios contain NaN")
            clipped_ratios = tf.clip_by_value(ratios, 1.0 - self.epsilon, 1.0 + self.epsilon)
            surrogate1 = ratios * advantages
            surrogate2 = clipped_ratios * advantages
            actor_loss = -tf.reduce_mean(tf.minimum(surrogate1, surrogate2))
            # print(actor_loss)
            entropy_bonus = tf.reduce_mean(entropy)
            # actor_loss_full = actor_loss 

            actor_loss_full = actor_loss - entropy_coef * entropy_bonus

        actor_grads = tape.gradient(actor_loss_full, self.actors[agent_ind].trainable_variables)
        # actor_grads = [tf.clip_by_norm(g, 0.5) for g in actor_grads]

        self.actor_optimizers[agent_ind].apply_gradients(zip(actor_grads, self.actors[agent_ind].trainable_variables))
        with self.summary_writers[agent_ind].as_default():
            summary.scalar('Actor Loss', actor_loss.numpy(), step=self.global_step)
            for idx, grad in enumerate(actor_grads):
                summary.histogram(f'Actor_Gradients/Layer_{idx}', grad, step=self.global_step)
            summary.scalar('Entropy', tf.reduce_mean(entropy).numpy(), step=self.global_step)
            summary.scalar('Learning Rate', self.actor_optimizers[agent_ind].learning_rate.numpy(), step=self.global_step)
            summary.histogram('Actions', actions, step=self.global_step)
            summary.histogram('Advantages', advantages, step=self.global_step)
    def critic_update(self, states, returns):
        # === Critic update ===
        returns = tf.convert_to_tensor(returns, dtype=tf.float32)

        with tf.GradientTape() as tape:
            
            values = self.critic.call(states, training = True)  # shape (batch, 1)
            critic_loss = tf.reduce_mean(tf.square(returns - values))

        critic_grads = tape.gradient(critic_loss, self.critic.trainable_variables)
        # critic_grads = [tf.clip_by_norm(g, 0.5) for g in critic_grads]

        self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))
            # summary.scalar('Critic Loss', critic_loss.numpy(), step=self.global_step)
            # for idx, grad in enumerate(critic_grads):
            #     summary.histogram(f'Critic_Gradients/Layer_{idx}', grad, step=self.global_step)
          # Добавляем логирование метрик


    def update_entropy_coef(self, episode, max_episodes):   
        entropy_coef = self.max_entropy * (0.5 ** (episode / (max_episodes/2)))
        return np.clip(entropy_coef, self.min_entropy, self.max_entropy)
    
    def save_models(self):
        for i in range(self.num_robots):
            self.actors[i].save(f'ppo_actor_robot_{i}.keras')
        print("\nModels saved successfully!")

    def train(self, max_episodes=500, batch_size=32, num_robots=2):
        all_rewards = []
        episodes_x, avg_y = [], []
        window = 5
        for episode in range(max_episodes):
            print('----------------------------------------------------------------------------')
            print("Episode: ",episode + 1)
            print('----------------------------------------------------------------------------')

            states = self.env.reset()
            states = [self.state_filter(state) for state in states]

            episode_rewards = [0] * num_robots
            done = False

            batch_data = {i: {
                'states': [], 
                'actions': [], 
                'rewards': [],
                'dones': [], 
                'raw_actions': [], 
                'log_probs': []
            } for i in range(num_robots)}
            critic_values = []
            step = 0

            while not done:
                actions, log_probs, _,_, raw_actions = self.get_action(states)
                # logger.info(f'Action:  {action}')
                # logger.info(f'Prob:  {prob}')
                next_states, rewards, dones, _ = self.env.step(actions)
                # rewards = [self.reward_filter(reward) for reward in rewards]

                next_states = [self.state_filter(state) for state in next_states]

                
                # if isinstance(dones, bool):
                #     dones = [dones] * self.num_robots
                if np.isnan(next_states).any():
                    print("Обнаружен NaN в состоянии!")
                    break

                critic_states = np.hstack(states)
                
                next_states = [np.reshape(next_state, [1, self.state_dim])for next_state in next_states]
                values_learned = float(self.critic.call(critic_states)[0, 0].numpy())

                for i in range(self.num_robots):
                    batch_data[i]['states'].append(states[i])
                    batch_data[i]['actions'].append(actions[i])
                    batch_data[i]['rewards'].append(rewards[i])
                    batch_data[i]['dones'].append(dones[i])
                    batch_data[i]['log_probs'].append(log_probs[i])
                    batch_data[i]['raw_actions'].append(raw_actions[i])
                    critic_values.append(values_learned) 
                    episode_rewards[i] += rewards[i]
                    done = any(dones) or (step >= self.env.max_steps)
                
                states = next_states
                step += 1

            if done or step % batch_size == 0:
                for i in range(self.num_robots):
                    critic_states = np.hstack(states)
                    next_values_learned = float(self.critic.call(critic_states)[0, 0].numpy()) 
                    critic_values.append(next_values_learned)
                    
                advantages, returns = self.compute_advantages(
                    [batch_data[i]['rewards'] for i in range(self.num_robots)],
                    critic_values,
                    [batch_data[i]['dones'] for i in range(self.num_robots)],
                )
                entropy_coef = self.update_entropy_coef(episode, max_episodes)
                # entropy_coef = 0.1

                for i in range(self.num_robots):
                    batch_data[i]['advantages'] = advantages
                    batch_data[i]['returns'] = returns
                    self.update(
                        np.vstack(batch_data[i]['states']),
                        batch_data[i]['actions'],
                        batch_data[i]['advantages'],
                        batch_data[i]['returns'],
                        batch_data[i]['log_probs'],
                        entropy_coef,
                        batch_data[i]['raw_actions'],
                        i)
                self.critic_update(critic_states, returns)
                for i in range(self.num_robots):
                    if episode == 0:
                        self.best_models[i] = episode_rewards[i]
                    if episode_rewards[i] > self.best_models[i]:
                        print(f"prev reward: {self.best_models[i]}, new reward {episode_rewards[i]}")
                        self.actors[i].save(f'ppo_actor_tb_{i}_best.keras')
                        print(' ...saving model... ')
                        self.best_models[i] = episode_rewards[i]



                for i in range(self.num_robots):
                    batch_data[i] = {k: [] for k in batch_data[i]}

                if done:
                    step = 0
                #     break


            all_rewards.append(episode_rewards)
            avg_reward = np.mean(episode_rewards)
            print(f"Episode {episode+1}/{max_episodes}")
            print(f"  Robot 1 Reward: {episode_rewards[0]:.2f}")
            print(f"  Robot 2 Reward: {episode_rewards[1]:.2f}")
            print(f"  Average Reward: {avg_reward:.2f}")

            with summary.create_file_writer(self.log_dir).as_default():
                summary.scalar('Average Reward', avg_reward, step=episode)
                summary.scalar('Max Reward', np.max(episode_rewards), step=episode)
                summary.scalar('Min Reward', np.min(episode_rewards), step=episode)
            
            for i in range(self.num_robots):
                with self.summary_writers[i].as_default():
                    summary.scalar('Total Reward', episode_rewards[i], step=episode)
                    summary.scalar('Steps', step, step=episode)
            
            self.global_step += 1

            # 2) онлайн-график (matplotlib)
            if (episode + 1) % window == 0:
                episodes_x.append(episode+1)
                avg_y.append(avg_reward)
                plt.clf()
                plt.plot(episodes_x, avg_y)
                plt.xlabel("Episode")
                plt.ylabel(f"Avg Reward ({window})")
                plt.title("Training Progress")
                plt.pause(0.01)
                plt.savefig(os.path.join('logs', 'train_decentr'))

            print(f'Episode {episode + 1}, Reward: {avg_reward}')


        for i in range(self.num_robots):
            self.actors[i].save(f'ppo_actor_robot_{i}.keras')
        for writer in self.summary_writers.values():
            writer.close()

        return all_rewards

def shutdown_handler(signum, frame):
    print("\nЗавершение симуляции Gazebo...")
    subprocess.run(['pkill', '-f', 'gazebo'])
    subprocess.run(['pkill', '-f', 'ros2'])
    sys.exit(0)

def main(args=None):
    rclpy.init(args=args)
    env = TurtleBotEnv()
    agent = PPOAgent(env)
    try:
        agent.train()
    except Exception as e:  
        print(f"\nCritical error occurred: {str(e)}", file=sys.stderr)
        traceback.print_exc()  
        # shutdown_handler(None, None)

    finally: 
        rclpy.shutdown()
        # shutdown_handler(None, None)

if __name__ == '__main__':
    main()