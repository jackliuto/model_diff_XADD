
import logging

import torch

from pyRDDLGym import RDDLEnv
from pyRDDLGym import ExampleManager
from pyRDDLGym.XADD.RDDLModelXADD import RDDLModelWXADD
from policy_learning.DQN.agents import DQN_Agent
from utils.dqn_utils import *

from pyRDDLGym.Policies.Agents import RandomAgent



params = Params("./params/dqn_params_inventory.json")
print('-----------------------------')
print(params.model_version, params.agent_type)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

DOMAIN_PATH = params.domain_path
INSTANCE_PATH = params.instance_path

myEnv = RDDLEnv.RDDLEnv(domain=DOMAIN_PATH, instance=INSTANCE_PATH)
model, context = get_xadd_model_from_file(f_domain=DOMAIN_PATH, f_instance=INSTANCE_PATH)

agent = RandomAgent(action_space=myEnv.action_space, num_actions=myEnv.numConcurrentActions)

total_reward = 0
state = myEnv.reset()
for step in range(myEnv.horizon):
    action = agent.sample_action()
    next_state, reward, done, info = myEnv.step(action)
    total_reward += reward
    print()
    print('step       = {}'.format(step))
    print('state      = {}'.format(state))
    print('action     = {}'.format(action))
    print('next state = {}'.format(next_state))
    print('reward     = {}'.format(reward))
    state = next_state
    if done:
        break
print("episode ended with reward {}".format(total_reward))
myEnv.close()