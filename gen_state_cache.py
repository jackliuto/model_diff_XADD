
import logging

import torch

from pyRDDLGym import RDDLEnv
from pyRDDLGym import ExampleManager
from pyRDDLGym.XADD.RDDLModelXADD import RDDLModelWXADD
from policy_learning.DQN.agents import DQN_Agent
from utils.dqn_utils import *



params = Params("./params/cache_params_reservoir.json")
print('-----------------------------')
print(params.agent_type)

params.domain_path = params.rddl_path + 'domain.rddl'
params.instance_path = params.rddl_path + "instance_{}_target.rddl".format(str(params.num_agent)+params.agent_name)
params.value_xadd_path: {"v_source":params.load_xadd_path+"{}/{}_steps/v_source.json".format(str(params.num_agent)+params.agent_name, params.num_xadd_steps), 
                        "v_target":params.load_xadd_path+"{}/{}_steps/v_target.json".format(str(params.num_agent)+params.agent_name, params.num_xadd_steps),
                        "v_diff":params.load_xadd_path+"{}/{}_steps/v_diff.json".format(str(params.num_agent)+params.agent_name, params.num_xadd_steps),
                    }

raise ValueError

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

DOMAIN_PATH = params.domain_path
INSTANCE_PATH = params.instance_path

myEnv = RDDLEnv.RDDLEnv(domain=DOMAIN_PATH, instance=INSTANCE_PATH)
model, context = get_xadd_model_from_file(f_domain=DOMAIN_PATH, f_instance=INSTANCE_PATH)


agent = DQN_Agent(env=myEnv, model=model, context=context, value_xadd_path=params.value_xadd_path, q_xadd_path=params.q_xadd_path, replay_memory_size=params.replay_memory_size, 
                    batch_size=params.batch_size, gamma=params.gamma, learning_rate=params.learning_rate, update_rate=params.update_rate, 
                    seed=params.seed, device=device, agent_type=params.agent_type, domain_type=params.domain_type)

agent.gen_value_cache()

if params.save_cache:
  cache_path = pathlib.Path(params.save_path+'{}_step/'.format(params.horizon_length))
  cache_path.mkdir(parents=True, exist_ok=True)

  save_value_function(params.save_path+'{}_step/'.format(params.horizon_length), 'v_source', vid_1, context_1)
  save_value_function(params.save_path+'{}_step/'.format(params.horizon_length), 'v_target', vid_2, context_2)
  save_value_function(params.save_path+'{}_step/'.format(params.horizon_length), 'v_diff', vid_diff, context_diff)

  save_q_function(params.save_path+'{}_step/'.format(params.horizon_length), 'q_source', q_1, context_1)
  save_q_function(params.save_path+'{}_step/'.format(params.horizon_length), 'q_target', q_2, context_2)
  save_q_function(params.save_path+'{}_step/'.format(params.horizon_length), 'q_diff', q_diff, context_diff)

