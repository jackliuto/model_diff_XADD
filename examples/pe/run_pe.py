import argparse
from xaddpy.xadd.xadd import XADD

from modeldiff.core.mdp import MDP
from modeldiff.core.policy import Policy
from modeldiff.core.action import Action
from modeldiff.policy_evaluation.pe import PolicyEvaluation
from modeldiff.utils.utils import get_xadd_model_from_file
from modeldiff.core.parser import Parser


def create_policy(mdp: MDP, context: XADD) -> Policy:
    move_pos_x = context.import_xadd('examples/pe/ex/move_pos_x.xadd', locals=context._str_var_to_var)
    move_pos_y = context.import_xadd('examples/pe/ex/move_pos_y.xadd', locals=context._str_var_to_var)
    do_nothing = context.import_xadd('examples/pe/ex/do_nothing.xadd', locals=context._str_var_to_var)
    xadd_policy = {
        'move-pos-x___a1': move_pos_x,
        'move-pos-y___a1':move_pos_y,
        'do-nothing___a1': do_nothing
    }
    policy = Policy(mdp)
    policy_dict = {}
    for aname, action in mdp.actions.items():
        policy_dict[action] = xadd_policy[aname]
    policy.load_policy(policy_dict)
    return policy


def test(args: argparse.ArgumentParser):
    xadd_model_1, context = get_xadd_model_from_file(args.f_env1, args.f_inst1)
    xadd_model_2, _ = get_xadd_model_from_file(args.f_env2, args.f_inst2, context=context)

    parser = Parser()
    mdp = parser.parse(xadd_model_1, is_linear=True)
    policy = create_policy(mdp, context)

    pe = PolicyEvaluation(mdp, policy, args.iter)
    v = pe.solve()

    pe.print(v)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--f_env1", type=str, default="RDDL/Navigation_disc_goal_551010/domain.rddl")
    parser.add_argument("--f_inst1", type=str, default=None)
    parser.add_argument("--f_env2", type=str, default="RDDL/Navigation_disc_goal_771010/domain.rddl")
    parser.add_argument("--f_inst2", type=str, default=None)
    parser.add_argument("--iter", type=int, default=2)
    args = parser.parse_args()
    test(args)
