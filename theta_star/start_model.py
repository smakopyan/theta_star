import rclpy
import numpy as np
from tensorflow import keras
from turtlebot_env import TurtleBotEnv, Zfilter
from actor import ImprovedActor, ResBlock

actor = keras.models.load_model(
    'ppo_actor_best.keras',
    custom_objects={'ImprovedActor': ImprovedActor, 'ResBlock': ResBlock}
)
rclpy.init() 
env = TurtleBotEnv()
state_filter = Zfilter(prev_filter=None, shape=env.observation_space.shape[0], clip=1.0)

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

    while not any(dones):
        states = [state_filter(state) for state in states]
        combined_states = []
        for i in range(num_robots):
            agent_id = np.eye(num_robots)[i]
            combined_state = np.concatenate([states[i], agent_id], axis=-1)
            combined_states.append(combined_state)
        states = combined_states

        actions = []
        for i in range(num_robots):
            state = np.reshape(states[i], [1, env.observation_space.shape[0]+2])
            action_scaled, _, _, _, _ = actor(state)
            
            action = action_scaled.numpy()[0]
            actions.append(action)


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