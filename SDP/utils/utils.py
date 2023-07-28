import pathlib
from pathlib import Path
from typing import Optional, Union

from pyRDDLGym.Core.Grounder.RDDLGrounder import RDDLGrounder
from pyRDDLGym.Core.Parser.parser import RDDLParser
from pyRDDLGym.Core.Parser.RDDLReader import RDDLReader
from pyRDDLGym.Examples.ExampleManager import ExampleManager
from pyRDDLGym.XADD.RDDLModelXADD import RDDLModelWXADD
from xaddpy.xadd import XADD

import os
import shutil
import itertools


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

    # for s in xadd_model.states.keys():
    #     for i in xadd_model.interm.keys():
    #         r_node = 
    #         s_node = xadd_model._var_name_to_node_id[s]
    #         i_node = xadd_model._var_name_to_node_id[i]
    #         print(s, s_node)
    #         print(i, i_node)
    # print(xadd_model.interm)
    # print(xadd_model.states)
    # print(xadd_model._var_name_to_node_id)
    # raise ValueError

    context = xadd_model._context
    return xadd_model, context

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
