
from pyRDDLGym.Core.Grounder.RDDLGrounder import RDDLGrounder
from pyRDDLGym.Core.Parser.parser import RDDLParser
from pyRDDLGym.Core.Parser.RDDLReader import RDDLReader
from pyRDDLGym.XADD.RDDLModelXADD import RDDLModelWXADD
from xaddpy.xadd import XADD

from SDP.core.parser import Parser
from SDP.core.mdp import MDP
from SDP.core.policy import Policy
from SDP.policy_evaluation.pe import PolicyEvaluation
from SDP.value_iteration.vi import ValueIteration

## Global Vars for SDP
DISCOUNT = 0.9
SDP_STEPS = 5

# Domain/Instance Path
f_domain = './RDDL/reservoir/reservoir_disc/domain.rddl'
f_instance = './RDDL/reservoir/reservoir_disc/instance_1res_source.rddl'

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
xadd_model = RDDLModelWXADD(model)
xadd_model.compile(simulation=False)
context = xadd_model._context

### Policy PolicyEvaluation

# parse model  
parser = Parser()
mdp = parser.parse(model, is_linear=True, discount=DISCOUNT) ## SDP currently only handle linear cases

# need to definte a policy by a string or load from xadd file
policy_str_release_true =  "( [rlevel___t1 - 45 <= 0] ( [0] ) ( [1] ) )"
policy_str_release_false =  "( [rlevel___t1 - 45 <= 0] ( [1] ) ( [0] ) )"
# ## import using: 
# policy_str_release_true = context.import_xadd('release___t1_true.xadd', locals=context._str_var_to_var)

policy_id_release_true = context.import_xadd(xadd_str=policy_str_release_true)
policy_id_release_false = context.import_xadd(xadd_str=policy_str_release_false)

xadd_policy = {
    'rlease___t1': policy_id_release_true

}

# load policy to mdp class
policy = Policy(mdp)
policy_dict = {}
for aname, action in mdp.actions.items():
    policy_dict[action] = xadd_policy[aname]







