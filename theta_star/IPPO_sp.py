import numpy as np
import tensorflow as tf
from keras import layers
import rclpy
from turtlebot_env import TurtleBotEnv, Zfilter, RunningStat
from critic import ImprovedCritic, world_to_map
from actor import ImprovedActor
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt
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

        self.obs_dim = env.observation_space.shape[0]  
        self.agent_id_dim = self.num_robots 
        self.action_dim = env.action_space.shape[0]
        self.state_dim = self.obs_dim + self.agent_id_dim
        self.grid_map = env.grid_map
        # self.reward_filter = Zfilter(prev_filter=None, shape=(), center = False, clip=1.0)

        self.best_model = 0

        self.x_range = np.array(env.x_range, dtype=np.float32)  # Диапазон X как массив NumPy
        self.y_range = np.array(env.y_range, dtype=np.float32)  # Диапазон Y как массив NumPy
        
        # Коэффициенты
        self.gamma = 0.99  # коэффициент дисконтирования
        self.epsilon = 0.1 # параметр клиппинга
        self.actor_lr = 0.0003
        self.critic_lr = 0.0003
        self.gaelam = 0.95
        self.min_entropy = 0.0001
        self.max_entropy = 0.01

        # self.alpha = 0.1

        self.actor = ImprovedActor(self.state_dim, self.action_dim)
        self.critic = ImprovedCritic(self.state_dim, env.grid_map, env.robots[0].optimal_path)

        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=self.actor_lr) 
        
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
            combined_state = states[i]
            state_tensor = tf.convert_to_tensor(combined_state.reshape(1, -1), dtype=tf.float32)
            # print(combined_state)
            # print(combined_state.shape)
            action_raw, log_prob, entropy, std, raw_action = self.actor(state_tensor, raw_actions = None)
            action = action_raw.numpy().squeeze()
            r_action = raw_action.numpy().squeeze()
            actions.append(action)
            log_probs.append(log_prob.numpy().squeeze())
            entropys.append(entropy.numpy().squeeze())
            stds.append(std)
            r_actions.append(r_action)
        return actions, log_probs, entropys, stds, r_actions   

    def compute_advantages(self, rewards, values, dones):
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

        return advantages, returns 
    # Обновление политик
    def update(self, batch_data, entropy_coef, states):
        all_states = np.vstack(np.asarray(states))
        all_actions = np.vstack([data["actions"] for data in batch_data.values()])
        all_log_probs_old = np.vstack([data["log_probs"] for data in batch_data.values()])

        all_advantages = np.vstack([data["advantages"] for data in batch_data.values()])
        all_returns = np.vstack([data["returns"] for data in batch_data.values()])
        all_raw_actions = np.vstack([data["raw_actions"] for data in batch_data.values()])

        states = tf.convert_to_tensor(all_states, dtype=tf.float32)
        actions = tf.convert_to_tensor(all_actions, dtype=tf.float32)
        log_probs_old = tf.convert_to_tensor(all_log_probs_old, dtype=tf.float32)
        advantages = tf.convert_to_tensor(all_advantages, dtype=tf.float32)
        returns = tf.convert_to_tensor(all_returns, dtype=tf.float32)
        raw_actions = tf.convert_to_tensor(all_raw_actions, dtype=tf.float32)

        # === Actor update ===
        with tf.GradientTape() as tape:
            log_probs, entropy = self.actor(states, training = True, raw_actions = raw_actions)
            log_probs = tf.debugging.check_numerics(log_probs, message="log_probs contain NaN")
            log_probs = tf.reshape(log_probs, (2, all_log_probs_old.shape[1]))
            print(log_probs.shape)

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

        actor_grads = tape.gradient(actor_loss_full, self.actor.trainable_variables)
        # actor_grads = [tf.clip_by_norm(g, 0.5) for g in actor_grads]

        self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))
        # === Critic update ===
        with tf.GradientTape() as tape:
            
            values = self.critic.call(states, training = True)  # shape (batch, 1)
            values = tf.reshape(values, (2, all_log_probs_old.shape[1]))

            critic_loss = tf.reduce_mean(tf.square(returns - values))

        critic_grads = tape.gradient(critic_loss, self.critic.trainable_variables)
        # critic_grads = [tf.clip_by_norm(g, 0.5) for g in critic_grads]

        self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))

        with summary.create_file_writer(self.log_dir).as_default():
            summary.scalar('Actor Loss', actor_loss.numpy(), step=self.global_step)
            for idx, grad in enumerate(actor_grads):
                summary.histogram(f'Actor_Gradients/Layer_{idx}', grad, step=self.global_step)
            summary.scalar('Critic Loss', critic_loss.numpy(), step=self.global_step)
            for idx, grad in enumerate(critic_grads):
                summary.histogram(f'Critic_Gradients/Layer_{idx}', grad, step=self.global_step)
            summary.scalar('Entropy', tf.reduce_mean(entropy).numpy(), step=self.global_step)
            summary.scalar('Learning Rate', self.actor_optimizer.learning_rate.numpy(), step=self.global_step)
            summary.histogram('Actions', actions, step=self.global_step)
            summary.histogram('Advantages', advantages, step=self.global_step)

    def update_entropy_coef(self, episode, max_episodes):   
        entropy_coef = self.max_entropy * (0.5 ** (episode / (max_episodes/2)))
        return np.clip(entropy_coef, self.min_entropy, self.max_entropy)
    
    def save_models(self):
        self.actor.save('ppo_actor_last.keras')
        print("\nModels saved successfully!")

    def train(self, max_episodes=500, batch_size=32, num_robots=2):
        all_rewards = []
        episodes_x, avg_y = [], []
        window = 5
        for episode in range(max_episodes):
            print('-------------------------------------------------------------')
            print("Episode: ",episode + 1)
            print('-------------------------------------------------------------')

            states = self.env.reset()

            episode_rewards = [0] * num_robots
            done = False

            batch_data = {i: {
                'states': [], 
                'actions': [], 
                'rewards': [],
                'dones': [], 
                'values_learned': [], 
                'raw_actions': [], 
                'log_probs': [],
            } for i in range(num_robots)}
            
            step = 0
            cmbd_states = []
            while not done:
                states_with_id = []
                if len(states[0]) == self.env.observation_space.shape[0]:
                #     states = [np.concatenate([state, [0.0, 0.0]], axis=-1) for state in states]
                    states = [self.state_filter(state) for state in states]
                for i in range(self.num_robots):
                    agent_id = np.eye(self.num_robots)[i]
                    state_with_id = np.concatenate([states[i], agent_id], axis=-1)
                    states_with_id.append(state_with_id)
                new_states = states_with_id


                actions, log_probs, _,_, raw_actions = self.get_action(new_states)
                next_states, rewards, dones, _ = self.env.step(actions)


                # next_states = [np.concatenate([next_states[i], actions[::-1][i]], axis=-1) for i in range(len(actions))]
                next_states = [self.state_filter(state) for state in next_states]

                
                # if isinstance(dones, bool):
                #     dones = [dones] * self.num_robots
                if np.isnan(next_states).any():
                    self.save_models()
                    print("Обнаружен NaN в состоянии!")
                    break
                next_states_with_id = []
                for i in range(self.num_robots):
                    agent_id = np.eye(self.num_robots)[i]
                    next_state_with_id = np.concatenate([next_states[i], agent_id], axis=-1)
                    next_states_with_id.append(next_state_with_id)
                
                next_states_with_id = [np.asarray(next_state).reshape(-1) for next_state in next_states_with_id]

                # next_states = [np.reshape(next_state, [1, self.state_dim-2])for next_state in next_states]
                next_states = [np.asarray(next_state).reshape(-1) for next_state in next_states]

                
                values_learned = [float(self.critic.call(state)[0, 0].numpy()) for i, state in enumerate(next_states_with_id)]

                for i in range(self.num_robots):
                    batch_data[i]['states'].append(states[i])
                    batch_data[i]['actions'].append(actions[i])
                    batch_data[i]['rewards'].append(rewards[i])
                    batch_data[i]['dones'].append(dones[i])
                    batch_data[i]['values_learned'].append(values_learned[i])
                    batch_data[i]['log_probs'].append(log_probs[i])
                    batch_data[i]['raw_actions'].append(raw_actions[i])
                    episode_rewards[i] += rewards[i]
                    
                    done = any(dones) or (step >= self.env.max_steps)
                
                states = next_states
                cmbd_states.append(next_states_with_id)
                step += 1

            if done or step % batch_size == 0:
                for i in range(self.num_robots):
                    next_values_learned = float(self.critic.call(next_states_with_id[i])[0, 0].numpy()) 
                    batch_data[i]['values_learned'].append(next_values_learned)
                    
                advantages, returns = self.compute_advantages(
                    [batch_data[i]['rewards'] for i in range(self.num_robots)],
                    [batch_data[i]['values_learned'] for i in range(self.num_robots)],
                    [batch_data[i]['dones'] for i in range(self.num_robots)]
                )

                entropy_coef = self.update_entropy_coef(episode, max_episodes)

                for i in range(self.num_robots):
                    batch_data[i]['advantages'] = advantages[i]
                    batch_data[i]['returns'] = returns[i]
                
                self.update(
                    batch_data,
                    entropy_coef,
                    cmbd_states
                    )
                avg_reward = np.mean(episode_rewards)
                if episode == 0:
                    self.best_model = avg_reward

                elif avg_reward > self.best_model and episode_rewards[0] > 0. and episode_rewards[1] > 0.:
                    print(f"prev reward: {self.best_model}, new reward {avg_reward}")
                    self.actor.save(f'ppo_actor_best.keras')
                    print(' ...saving model... ')
                    self.best_model = avg_reward



                # for i in range(self.num_robots):
                #     batch_data[i] = {k: [] for k in batch_data[i]}

                if done:
                    step = 0
                #     break


            all_rewards.append(episode_rewards)
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
                plt.savefig(os.path.join('logs', 'train_CTDE'))

            print(f'Episode {episode + 1}, Reward: {avg_reward}')


        self.actor.save(f'ppo_actor.keras')
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