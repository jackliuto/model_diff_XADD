
import logging

import torch

from pyRDDLGym import RDDLEnv
from pyRDDLGym import ExampleManager
from pyRDDLGym.XADD.RDDLModelXADD import RDDLModelWXADD
from policy_learning.DQN.agents import DQN_Agent
from utils.dqn_utils import *
from utils.xadd_utils import *

import multiprocessing

params = Params("./params/cache_params_reservoir.json")
print('-----------------------------')
print(params.domain_type)

def gen_xadd_cache(num_xadd_steps):
  print('{} cache xadd horizon is porcessing.......'.format(num_xadd_steps))
  params.domain_path = params.rddl_path + 'domain.rddl'
  params.instance_path = params.rddl_path + "instance_{}_target.rddl".format(str(params.num_agent)+params.agent_name)
  params.value_xadd_path = {"v_source":params.load_xadd_path+"{}/{}_step/v_source.json".format(str(params.num_agent)+params.agent_name, num_xadd_steps), 
                          "v_target":params.load_xadd_path+"{}/{}_step/v_target.json".format(str(params.num_agent)+params.agent_name, num_xadd_steps),
                          "v_diff":params.load_xadd_path+"{}/{}_step/v_diff.json".format(str(params.num_agent)+params.agent_name, num_xadd_steps),
                      }
  params.q_xadd_path = {"q_source":params.load_xadd_path+"{}/{}_step/q_source.json".format(str(params.num_agent)+params.agent_name, num_xadd_steps), 
                        "q_target":params.load_xadd_path+"{}/{}_step/q_target.json".format(str(params.num_agent)+params.agent_name, num_xadd_steps),
                        "q_diff":params.load_xadd_path+"{}/{}_step/q_diff.json".format(str(params.num_agent)+params.agent_name, num_xadd_steps),
                      }


  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

  DOMAIN_PATH = params.domain_path
  INSTANCE_PATH = params.instance_path

  myEnv = RDDLEnv.RDDLEnv(domain=DOMAIN_PATH, instance=INSTANCE_PATH)
  model, context = get_xadd_model_from_file(f_domain=DOMAIN_PATH, f_instance=INSTANCE_PATH)


  agent = DQN_Agent(env=myEnv, model=model, context=context, value_xadd_path=params.value_xadd_path, q_xadd_path=params.q_xadd_path, domain_type=params.domain_type)

  value_xadd_cache, q_xadd_cache = agent.gen_value_cache()

  if params.save_cache:
    save_dir = params.save_cache_path+'{}/{}_step/'.format(str(params.num_agent)+params.agent_name, num_xadd_steps)
    cache_path = pathlib.Path(save_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    save_cache(save_dir, value_xadd_cache)
    save_cache(save_dir, q_xadd_cache)
  print('{} cache xadd horizon is completed'.format(num_xadd_steps))

def main():
  num_processes = multiprocessing.cpu_count()

  pool = multiprocessing.Pool(processes=num_processes)

  results = pool.map(gen_xadd_cache, params.num_xadd_steps)

  pool.close()

  pool.join()

  print('all jobs completed')


main()