import pathlib
from pathlib import Path
from typing import Optional, Union

from pyRDDLGym.Core.Grounder.RDDLGrounder import RDDLGrounder
from pyRDDLGym.Core.Parser.parser import RDDLParser
from pyRDDLGym.Core.Parser.RDDLReader import RDDLReader
from pyRDDLGym.Examples.ExampleManager import ExampleManager
from pyRDDLGym.XADD.RDDLModelXADD import RDDLModelWXADD
from xaddpy.xadd import XADD


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
    xadd_model.compile()
    context = xadd_model._context
    return xadd_model, context
