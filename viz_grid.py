
import logging

import random

import os

import torch

import matplotlib.pyplot as plt
import numpy as np


from pyRDDLGym import RDDLEnv
from pyRDDLGym import ExampleManager
from pyRDDLGym.XADD.RDDLModelXADD import RDDLModelWXADD
from policy_learning.DQN.agents import DQN_Agent
from utils.dqn_utils import *


params = Params("./params/dqn_params_navigation.json")
print('-----------------------------')
print(params.agent_type)

params.domain_path = params.rddl_path + 'domain.rddl'

params.instance_path = params.rddl_path + "instance_{}_target.rddl".format(str(params.num_agent)+params.agent_name)
params.value_xadd_path = {"v_source":params.load_xadd_path+"{}/{}_step/v_source.json".format(str(params.num_agent)+params.agent_name, params.num_xadd_steps), 
                        "v_target":params.load_xadd_path+"{}/{}_step/v_target.json".format(str(params.num_agent)+params.agent_name, params.num_xadd_steps),
                        "v_diff":params.load_xadd_path+"{}/{}_step/v_diff.json".format(str(params.num_agent)+params.agent_name, params.num_xadd_steps),
                    }
params.q_xadd_path = {"q_source":params.load_xadd_path+"{}/{}_step/q_source.json".format(str(params.num_agent)+params.agent_name, params.num_xadd_steps), 
                      "q_target":params.load_xadd_path+"{}/{}_step/q_target.json".format(str(params.num_agent)+params.agent_name, params.num_xadd_steps),
                      "q_diff":params.load_xadd_path+"{}/{}_step/q_diff.json".format(str(params.num_agent)+params.agent_name, params.num_xadd_steps),
                    }

params.value_cache_path = {"v_source":params.load_cache_path+"{}/{}_step/v_source.json".format(str(params.num_agent)+params.agent_name, params.num_xadd_steps), 
                        "v_target":params.load_cache_path+"{}/{}_step/v_target.json".format(str(params.num_agent)+params.agent_name, params.num_xadd_steps),
                        "v_diff":params.load_cache_path+"{}/{}_step/v_diff.json".format(str(params.num_agent)+params.agent_name, params.num_xadd_steps),
                    }
params.q_cache_path = {"q_source":params.load_cache_path+"{}/{}_step/q_source.json".format(str(params.num_agent)+params.agent_name, params.num_xadd_steps), 
                      "q_target":params.load_cache_path+"{}/{}_step/q_target.json".format(str(params.num_agent)+params.agent_name, params.num_xadd_steps),
                      "q_diff":params.load_cache_path+"{}/{}_step/q_diff.json".format(str(params.num_agent)+params.agent_name, params.num_xadd_steps),
                    }


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

DOMAIN_PATH = params.domain_path
INSTANCE_PATH = params.instance_path

results_dict = {'train_reward':[], 'eval_reward':[], 'params':params.params}

myEnv = RDDLEnv.RDDLEnv(domain=DOMAIN_PATH, instance=INSTANCE_PATH)
model, context = get_xadd_model_from_file(f_domain=DOMAIN_PATH, f_instance=INSTANCE_PATH)


agent = DQN_Agent(env=myEnv, model=model, context=context, 
                value_xadd_path=params.value_xadd_path, q_xadd_path=params.q_xadd_path,
                value_cache_path=params.value_cache_path, q_cache_path=params.q_cache_path, 
                replay_memory_size=params.replay_memory_size, batch_size=params.batch_size, 
                gamma=params.gamma, learning_rate=params.learning_rate, update_rate=params.update_rate, tau=params.tau, 
                    seed=params.seed, device=device, agent_type=params.agent_type, use_cache=params.use_cache)

print(agent.state_index_dict)

value_dict = agent.value_cache
q_dict = agent.q_cache

viz_source = np.zeros((11,11))
viz_target = np.zeros((11,11))
viz_diff = np.zeros((11,11))

print(agent.state_index_dict)

np.set_printoptions(precision=2)

print()

for i in range(11):
    for j in range(11):
        coor_str = '({}, {})'.format(j,i)
        v_source = float(value_dict['v_source'][coor_str])
        v_target = float(value_dict['v_target'][coor_str])
        v_diff = float(value_dict['v_diff'][coor_str])
        viz_source[i][j] = v_source
        viz_target[i][j] = v_target
        viz_diff[i][j] = v_diff

print(viz_source[:, 1:])
print(viz_target[:, 1:])
print(viz_diff[:, 1:])

viz_lb = viz_source+ viz_diff

print(viz_lb[:, 1:])
        

fig, ax = plt.subplots()

im = ax.imshow(viz_lb[:, 1:], cmap='gray', interpolation='nearest')
ax.set_title(f'V_lb 20 steps')
ax.set_xlabel('x')
ax.set_ylabel('y')
fig.colorbar(im)
ax.set_xticks(np.arange(10))
ax.set_yticks(np.arange(10))

# plt.show()

plt.savefig(f"./v_lb.png")
