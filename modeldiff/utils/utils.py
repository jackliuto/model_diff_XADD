
from pathlib import Path
from typing import Optional

from pyRDDLGym.Core.Grounder.RDDLGrounder import RDDLGrounder
from pyRDDLGym.Core.Parser.parser import RDDLParser
from pyRDDLGym.Core.Parser.RDDLReader import RDDLReader
from pyRDDLGym.Examples.ExampleManager import ExampleManager
from pyRDDLGym.XADD.RDDLModelXADD import RDDLModelWXADD
from xaddpy.xadd import XADD


# gen xadd model from a RDDLEnv
def get_xadd_model_from_file(env_name: str, instance: int = 0, context: XADD = None):
    env_info = ExampleManager.GetEnvInfo(env_name)
    domain = env_info.get_domain()
    instance = env_info.get_instance(instance)
    # Read and parse domain and instance
    reader = RDDLReader(domain, instance)
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
    xadd_model.compile()
    context = xadd_model._context
    return xadd_model, context

