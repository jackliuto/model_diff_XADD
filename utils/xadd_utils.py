import json
import logging
import os
import shutil
import itertools

import numpy as np

import pathlib
from pathlib import Path
from typing import Optional, Union

from pyRDDLGym.Core.Grounder.RDDLGrounder import RDDLGrounder
from pyRDDLGym.Core.Parser.parser import RDDLParser
from pyRDDLGym.Core.Parser.RDDLReader import RDDLReader
from pyRDDLGym.Examples.ExampleManager import ExampleManager
from pyRDDLGym.XADD.RDDLModelXADD import RDDLModelWXADD
from xaddpy.xadd import XADD

class Params():
    """Class that loads hyperparameters from a json file.

    Example:
    ```
    params = Params(json_path)
    print(params.learning_rate)
    params.learning_rate = 0.5  # change the value of learning_rate in params
    ```
    """

    def __init__(self, json_path):
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)
            self.params = params

    def save(self, json_path):
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)
            
    def update(self, json_path):
        """Loads parameters from json file"""
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        """Gives dict-like access to Params instance by `params.dict['learning_rate']"""
        return self.__dict__


class RunningAverage():
    """A simple class that maintains the running average of a quantity
    
    Example:
    ```
    loss_avg = RunningAverage()
    loss_avg.update(2)
    loss_avg.update(4)
    loss_avg() = 3
    ```
    """
    def __init__(self):
        self.steps = 0
        self.total = 0
    
    def update(self, val):
        self.total += val
        self.steps += 1
    
    def __call__(self):
        return self.total/float(self.steps)
        
    
def set_logger(log_path):
    """Set the logger to log info in terminal and file `log_path`.

    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to `model_dir/train.log`.

    Example:
    ```
    logging.info("Starting training...")
    ```

    Args:
        log_path: (string) where to log
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)


def save_dict_to_json(d, json_path):
    """Saves dict of floats in json file

    Args:
        d: (dict) of float-castable values (np.float, int, float, etc.)
        json_path: (string) path to json file
    """
    with open(json_path, 'w') as f:
        # We need to convert the values to float for json (it doesn't accept np.array, np.float, )
        d = {k: float(v) for k, v in d.items()}
        json.dump(d, f, indent=4)


def save_checkpoint(state, is_best, checkpoint):
    """Saves model and training parameters at checkpoint + 'last.pth.tar'. If is_best==True, also saves
    checkpoint + 'best.pth.tar'

    Args:
        state: (dict) contains model's state_dict, may contain other keys such as epoch, optimizer state_dict
        is_best: (bool) True if it is the best model seen till now
        checkpoint: (string) folder where parameters are to be saved
    """
    filepath = os.path.join(checkpoint, 'last.pth.tar')
    if not os.path.exists(checkpoint):
        print("Checkpoint Directory does not exist! Making directory {}".format(checkpoint))
        os.mkdir(checkpoint)
    else:
        print("Checkpoint Directory exists! ")
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'best.pth.tar'))


def load_checkpoint(checkpoint, model, optimizer=None):
    """Loads model parameters (state_dict) from file_path. If optimizer is provided, loads state_dict of
    optimizer assuming it is present in checkpoint.

    Args:
        checkpoint: (string) filename which needs to be loaded
        model: (torch.nn.Module) model for which the parameters are loaded
        optimizer: (torch.optim) optional: resume optimizer from checkpoint
    """
    if not os.path.exists(checkpoint):
        raise("File doesn't exist {}".format(checkpoint))
    checkpoint = torch.load(checkpoint)
    model.load_state_dict(checkpoint['state_dict'])

    if optimizer:
        optimizer.load_state_dict(checkpoint['optim_dict'])

    return checkpoint


def state_to_vec(state):

    keys = list(state.keys())
    keys.sort()
    state_array = np.array([state[key] for key in keys])

    return state_array

def gen_action_space(action_name_list):
    bool_combos = [list(i) for i in itertools.product([0, 1], repeat=len(action_name_list))]
    action_list = []
    for b in bool_combos:
        a = {}
        for i, v in enumerate(b):
            a[action_name_list[i]] = True if v==1 else False
        action_list.append(a)
    return action_list

# gen xadd model from a RDDLEnv
def get_xadd_model_from_file(
        f_domain: Union[str, pathlib.PosixPath],
        f_instance: Union[str, pathlib.PosixPath] = None,
        context: XADD = None
):
    if f_instance is None:
        f_instance = str(f_domain).replace('domain.rddl', 'instance0.rddl')
    
    # Read and parse domain and instance
    reader = RDDLReader(f_domain, f_instance)
    domain = reader.rddltxt
    parser = RDDLParser(None, False)
    parser.build()

    # Parse RDDL file
    rddl_ast = parser.parse(domain)

    # Ground domain
    grounder = RDDLGrounder(rddl_ast)
    model = grounder.Ground()

    # XADD compilation
    xadd_model = RDDLModelWXADD(model, context=context)
    xadd_model.compile(simulation=False)
    context = xadd_model._context
    return xadd_model, context


def save_value_function(fpath, name, node_id, context):
    node = context._id_to_node.get(node_id, None)
    node.turn_off_print_node_info()
    node_str = str(node)
    node_dict = {name:node_str}
    with open(fpath+name+'.json', 'w') as f:
        json.dump(node_dict, f)

def save_q_function(fpath, name, q_dict, context):
    q_xadd_dict = {}
    for k, v in q_dict.items():
        action = k
        node_id = v
        node = context._id_to_node.get(node_id, None)
        node.turn_off_print_node_info()
        node_str = str(node)
        q_xadd_dict[action] = node_str
    with open(fpath+name+'.json', 'w') as f:
        json.dump(q_xadd_dict, f)


def save_cache(fpath, cache_dict):
    for k,v in cache_dict.items():
        with open(fpath+k+'.json', 'w') as f:
            json.dump(v, f)
