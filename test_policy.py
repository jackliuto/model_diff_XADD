import torch

from pyRDDLGym import RDDLEnv
from policy_learning.DQN.agents import DQN_Agent
from utils.dqn_utils import *

import numpy as np


params = Params("./params/test_policy_params_reservoir.json")
print('-----------------------------')


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

myEnv = RDDLEnv.RDDLEnv(domain=DOMAIN_PATH, instance=INSTANCE_PATH)
model, context = get_xadd_model_from_file(f_domain=DOMAIN_PATH, f_instance=INSTANCE_PATH)


agent = DQN_Agent(env=myEnv, model=model, context=context, 
                value_xadd_path=params.value_xadd_path, q_xadd_path=params.q_xadd_path,
                value_cache_path=params.value_cache_path, q_cache_path=params.q_cache_path, 
                use_cache=params.use_cache)


def value_cache_to_array(cache_dict, dim=(100)):
    value_dict = {}
    for v_name, v_dict in cache_dict.items():
        value_array = np.zeros(dim)
        for k,v in v_dict.items():
            index_tuple = tuple([int(i)-1 for i in list(k[1:-1].split(',')) if i != ''])
            value_array[index_tuple] = float(v)
        value_dict[v_name] = value_array
    return value_dict

def q_cache_to_array(cache_dict, dim=(100)):
    q_dict = {}
    for q_name, q_lst in cache_dict.items():
        new_q_lst = []
        for i in q_lst:
            action = i[0]
            value_array = np.zeros(dim)
            for k,v in i[1].items():
                index_tuple = tuple([int(i)-1 for i in list(k[1:-1].split(',')) if i != ''])
                value_array[index_tuple] = float(v)
            new_q_lst.append([action, value_array])
        q_dict[q_name] = new_q_lst
    return q_dict

def find_best_action(q_lst):
    print("----Action Codes:")
    for i,a in enumerate(q_lst):    
        print(i, a[0])
    states_len = len(q_lst[0][1])
    policy_lst = []
    for i in range(states_len):
        v_lst = []
        for j in q_lst:
            v_lst.append(j[1][i])
        policy_lst.append(list(np.where(v_lst == np.amax(v_lst))[0]))
    return policy_lst
    


value_dict = value_cache_to_array(agent.value_cache)
q_dict =  q_cache_to_array(agent.q_cache)


v_source_lst = value_dict["v_source"]
v_target_lst = value_dict["v_target"]
q_target_lst = q_dict['q_target']
q_source_lst = q_dict['q_source']

policy_q_source = find_best_action(q_source_lst)
policy_q_target = find_best_action(q_target_lst)

for i in range(len(policy_q_source)):
    print(i, policy_q_source[i], policy_q_target[i], v_source_lst[i], v_target_lst[i])
        

# for i in range(len(cache_array_v_target)):
#     print(i, round(cache_array_v_source[i],2), round(cache_array_v_target[i],2))




        