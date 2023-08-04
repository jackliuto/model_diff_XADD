
import logging

import torch

from pyRDDLGym import RDDLEnv
from pyRDDLGym import ExampleManager
from pyRDDLGym.XADD.RDDLModelXADD import RDDLModelWXADD
from policy_learning.DQN.agents import DQN_Agent
from utils.dqn_utils import *



params = Params("./params/dqn_params_reservoir.json")
print('-----------------------------')
print(params.model_version, params.agent_type)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

DOMAIN_PATH = params.domain_path
INSTANCE_PATH = params.instance_path

myEnv = RDDLEnv.RDDLEnv(domain=DOMAIN_PATH, instance=INSTANCE_PATH)
model, context = get_xadd_model_from_file(f_domain=DOMAIN_PATH, f_instance=INSTANCE_PATH)


agent = DQN_Agent(env=myEnv, model=model, context=context, value_xadd_path=params.value_xadd_path, q_xadd_path=params.q_xadd_path, replay_memory_size=params.replay_memory_size, 
                    batch_size=params.batch_size, gamma=params.gamma, learning_rate=params.learning_rate, update_rate=params.update_rate, 
                    seed=params.seed, device=device, agent_type=params.agent_type)

total_reward = 0

set_logger('train.log')


train_list = []
eval_list = []
for eps in range(params.num_episodes*2):
    state = myEnv.reset()
    step = 0
    eps_reward = 0
    if eps % 2 == 0:
        epsilon = params.epsilon
        psi = params.psi
        mode = 'train'
    else:
        epsilon = 0.0
        psi = 0.0
        mode = 'eval'

    while step < params.eps_length:
        step +=1
        # state_tensor = state_to_vec(state)
        state_vec, action_vec, action = agent.act(state, epsilon, psi)
        next_state, reward, done, info = myEnv.step(action)
        next_state_vec = agent.state_to_vec(next_state)
        done = int(False) ## no terminal state thus have done all set to false
        if mode == 'train':
            agent.step(state_vec, action_vec, reward, next_state_vec, done)
        eps_reward += reward*params.gamma**step
        # eps_reward += reward
        # print()
        # print('step       = {}'.format(step))
        # print('state      = {}'.format(state))
        # print('action     = {}'.format(action))
        # print('next state = {}'.format(next_state))
        # print('reward     = {}'.format(reward))
        state = next_state
    # raise ValueError
    if mode == 'eval':
        eval_list.append(eps_reward)
        total_reward += eps_reward
        print("episode {} ended with reward {}".format(eps, eps_reward))
    else:
        train_list.append(eps_reward)

print('total reward {}'.format(total_reward))

results_dict = {'train_reward':train_list, 'eval_reward':eval_list, 'params':params.params}

if params.save:
    with open('./results/{}_{}.json'.format(params.model_version, params.agent_type),'w') as f:
        json.dump(results_dict, f)

myEnv.close()