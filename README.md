# model_diff_XADD
Repo for XADD model-diff experiments

## Instructions for SDP

Example code in sdp_example.py

Requirement:
Install PyRDDLGYm and xaddpy from two repositories 

https://github.com/jihwan-jeong/xaddpy/tree/main/xaddpy \
https://github.com/ataitler/pyRDDLGym

    git clone https://github.com/jihwan-jeong/xaddpy/tree/main/xaddpy
    cd xaddpy
    pip install -e .

    git clone https://github.com/ataitler/pyRDDLGym
    cd pyRDDLGym
    pip install -e .

First you need to import neccesray libs

    from SDP.utils.utils import get_xadd_model_from_file
    from SDP.core.parser import Parser
    from SDP.core.mdp import MDP
    from SDP.core.policy import Policy
    from SDP.policy_evaluation.pe import PolicyEvaluation
    from SDP.value_iteration.vi import ValueIteration

Then set the discount rate, number of SDP steps and define the path for RDDL domain and instance

    ## Global Vars for SDP
    DISCOUNT = 0.9
    N_STEPS = 2

    # Domain/Instance Path
    f_domain = './RDDL/reservoir/reservoir_disc/domain.rddl'
    f_instance = './RDDL/reservoir/reservoir_disc/instance_1res_source.rddl'

Generate RDDL model and XADD context with the following code

    # load xadd model and context see SDP.utils for details
    model, context = get_xadd_model_from_file(f_domain, f_instance)



### Value Iteration
Value Iteration can be done using ValueIteration class and its solve() method \
The sovler returns the XADD node id for the value function after SDP, also a list of q-values in the format of [(action_dict, node_id), ....]


    parser = Parser()
    mdp = parser.parse(model, is_linear=True, discount=DISCOUNT) ## SDP currently only handle linear cases
    vi= ValueIteration(mdp, N_STEPS)
    value_id_vi, q_id_list_vi = vi.solve()

you can printout the string using
    vi.print(value_id_vi)

### Policy Evluation
Policy Evluation need a policy to be defined manually, you can either import an xadd file or generate a string, here is an example of a policy specified for the reservoir domain with 1 re

    ### Policy PolicyEvaluation
    parser = Parser()
    mdp = parser.parse(model, is_linear=True, discount=DISCOUNT) ## SDP currently only handle linear cases

    # need to definte a policy by a string or load from xadd file
    policy_str_release_true =  "( [rlevel___t1 - 55 <= 0] ( [0] ) ( [1] ) )"
    policy_str_release_false =  "( [rlevel___t1 - 55 <= 0] ( [1] ) ( [0] ) )"
    # ## import using: 
    # policy_str_release_true = context.import_xadd('release___t1_true.xadd', locals=context._str_var_to_var)

    # get node ids for xadd
    policy_id_release_true = context.import_xadd(xadd_str=policy_str_release_true)
    policy_id_release_false = context.import_xadd(xadd_str=policy_str_release_false)

    # make a dictionary of action as stringst to node id
    xadd_policy = {
        '{release___t1: True}': policy_id_release_true,
        '{release___t1: False}': policy_id_release_false,
    }

    # load policy to mdp class
    policy = Policy(mdp)
    policy_dict = {}
    for aname, action in mdp.actions.items():
        policy_dict[action] = xadd_policy[aname]
    policy.load_policy(policy_dict)


The solving step is similar to VI but with PolicyEvaluation class

    ## do policy evluation for n steps
    pe = PolicyEvaluation(mdp, policy, N_STEPS)
    value_id_pe, q_id_list_pe = pe.solve()

    # can printout value XADD using print function in pe
    print(pe.print(value_id_pe))
