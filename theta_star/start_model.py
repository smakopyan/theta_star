import rclpy
import numpy as np
from tensorflow import keras
from turtlebot_env import TurtleBotEnv
from actor import ImprovedActor

actor = [
    keras.models.load_model(
        'ppo_actor_robot_0.keras',
        custom_objects={'ImprovedActor': ImprovedActor}
    ),
    keras.models.load_model(
        'ppo_actor_robot_1.keras',
        custom_objects={'ImprovedActor': ImprovedActor}
    )
]
rclpy.init() 
env = TurtleBotEnv()

num_episodes = 10
total_rewards = []
total_steps = []
success_count = 0
num_robots = 2

for episode in range(num_episodes):
    states = env.reset()
    dones = [False] * num_robots
    episode_rewards = [0] * num_robots
    steps = 0

    while not all(dones):
        actions = []
        for i in range(num_robots):
            state = np.reshape(states[i], [1, env.observation_space.shape[0]])
            action_probs = actor[i](state)
            actions.append(np.argmax(action_probs))

        next_states, rewards, dones, info = env.step(actions)
        
        # Convert dones to a list if it's a boolean
        if isinstance(dones, bool):
            dones = [dones] * num_robots
        
        for i in range(num_robots):
            episode_rewards[i] += rewards[i]
            if dones[i] and (steps < env.max_steps):
                success_count += 1

        states = next_states
        steps += 1

    total_rewards.append(episode_rewards)
    total_steps.append(steps)
    print(f'Episode {episode+1}: Rewards = {episode_rewards}, Steps = {steps}')

env.close()

print("\n=== Evaluation Results ===")
print(f'✅ Средняя награда: {np.mean(total_rewards):.2f}')
print(f'✅ Среднее число шагов: {np.mean(total_steps):.2f}')
print(f'✅ Успешные эпизоды: {success_count} / {num_episodes} ({(success_count / num_episodes) * 100:.1f}%)')