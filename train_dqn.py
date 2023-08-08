
import logging

import random

import torch

from pyRDDLGym import RDDLEnv
from pyRDDLGym import ExampleManager
from pyRDDLGym.XADD.RDDLModelXADD import RDDLModelWXADD
from policy_learning.DQN.agents import DQN_Agent
from utils.dqn_utils import *


params = Params("./params/dqn_params_reservoir.json")
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
                gamma=params.gamma, learning_rate=params.learning_rate, update_rate=params.update_rate, 
                    seed=params.seed, device=device, agent_type=params.agent_type, use_cache=params.use_cache)

for i in range(params.num_runs):

    agent.reset(seed=i)

    total_reward = 0
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

    results_dict["train_reward"].append(train_list)
    results_dict["eval_reward"].append(eval_list)

    # ## this is for printing final results
    # for i in range(100):
    #     state = {'rlevel___t1':i}
    #     state_vec = agent.state_to_vec(state)
        
    #     state_tensor = torch.from_numpy(state_vec).float().unsqueeze(0).to(device)
    #     action_values = agent.network(state_tensor)
    #     print(state_tensor, action_values)


if params.save:
    with open('./results/{}_{}_{}steps_{}.json'.format(params.domain_type, str(params.num_agent)+params.agent_name, 
                                                    params.eps_length, params.agent_type),'w') as f:
        json.dump(results_dict, f)

myEnv.close()