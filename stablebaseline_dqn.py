

from pyRDDLGym import RDDLEnv

from policy_learning.envs.navEnv import envWrapperNav

from stable_baselines3 import DQN

DOMAIN_PATH = "./RDDL/navigation/navigation_disc/domain.rddl"
INSTANCE_PATH = "./RDDL/navigation/navigation_disc/instance_1agent_target.rddl"

RDDLEnv = RDDLEnv.RDDLEnv(domain=DOMAIN_PATH, instance=INSTANCE_PATH)

env = envWrapperNav(RDDLEnv)
env.max_episode_length = 1000

model = DQN('MultiInputPolicy', env, verbose=0)


total_timesteps = 10000    # Adjust as needed

model.learn(total_timesteps=total_timesteps)

# Access information from the model's info attribute
train_info = model.info

# Access specific information
total_timesteps = train_info['total_timesteps']
time_elapsed = train_info['time_elapsed']
num_episodes = train_info['num_episodes']
mean_reward = train_info['ep_rew_mean']

# Print or use the information as needed
print(f"Total Timesteps: {total_timesteps}")
print(f"Time Elapsed: {time_elapsed} seconds")
print(f"Number of Episodes: {num_episodes}")
print(f"Mean Reward: {mean_reward}")