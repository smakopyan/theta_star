import rclpy
import numpy as np
from tensorflow import keras
from env import TurtleBotEnv
from actor import ImprovedActor

# Загружаем обученную модель
actor = keras.models.load_model('ppo_turtlebot_actor.keras')
rclpy.init() 
# Создаём среду
env = TurtleBotEnv()

num_episodes = 10  # Количество тестовых запусков
total_rewards = []
total_steps = []
success_count = 0

for episode in range(num_episodes):
    state = env.reset()
    state = np.reshape(state, [1, env.observation_space.shape[0]])
    
    done = False
    episode_reward = 0
    steps = 0

    while not done:
        action_probs = actor(state)
        action = np.argmax(action_probs)  # Выбираем действие с наибольшей вероятностью

        next_state, reward, done, info = env.step(action)
        next_state = np.reshape(next_state, [1, env.observation_space.shape[0]])

        state = next_state
        episode_reward += reward
        steps += 1

        if "goal_reached" in info and info["goal_reached"]:
            success_count += 1

    total_rewards.append(episode_reward)
    total_steps.append(steps)

    print(f'Episode {episode + 1}: Reward = {episode_reward}, Steps = {steps}')

env.close()

# 🔹 Вывод метрик
print("\n=== Evaluation Results ===")
print(f'✅ Средняя награда: {np.mean(total_rewards):.2f}')
print(f'✅ Среднее число шагов: {np.mean(total_steps):.2f}')
print(f'✅ Успешные эпизоды: {success_count} / {num_episodes} ({(success_count / num_episodes) * 100:.1f}%)')