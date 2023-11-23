import warnings
warnings.filterwarnings("ignore")

from pyRDDLGym import RDDLEnv

from policy_learning.envs.navEnv import envWrapperNav

from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
from stable_baselines3.common.logger import configure

from stable_baselines3.common.env_checker import check_env

# globalvars
DOMAIN_PATH = "./RDDL/navigation/navigation_disc/domain.rddl"
INSTANCE_PATH = "./RDDL/navigation/navigation_disc/instance_1agent_source.rddl"

GAMMA = 0.9
TAU = 0.001
EXP_BEG = 0.9
EXP_FINAL = 0.1
BUFFER_SIZE = 100000
BATCH_SIZE = 128

total_timesteps = 10000000    # Adjust as needed


# set enviroment
RDDLEnv = RDDLEnv.RDDLEnv(domain=DOMAIN_PATH, instance=INSTANCE_PATH)
env = envWrapperNav(RDDLEnv, max_episode_length=1000)


check_env(env)

# set callbacks
checkpoint_callback = CheckpointCallback(save_freq = 1000, 
                                         save_path = '..//checkpoints/',
                                         name_prefix = 'rl_model')

# eval_callback = EvalCallback(eval_env, best_model_save_path='./logs/',
#                              log_path='./logs/', eval_freq=500,
#                              deterministic=True, render=False)

# set up logger
tmp_path = "./policy_learning/logs/"
new_logger = configure(tmp_path, ["stdout", "csv", "log", "tensorboard"])                    

model = DQN('MultiInputPolicy', env, 
            gamma=GAMMA,
            tau=TAU,
            exploration_initial_eps=EXP_BEG,
            exploration_final_eps=EXP_FINAL,
            buffer_size=BUFFER_SIZE,
            batch_size=BATCH_SIZE,
            verbose=1)

model.set_logger(new_logger)


training_info = model.learn(total_timesteps=total_timesteps,
                            callback=checkpoint_callback)

