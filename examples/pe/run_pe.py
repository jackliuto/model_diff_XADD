import argparse

from modeldiff.policy_evaluation.pe import PolicyEvaluation
from modeldiff.utils.utils import get_xadd_model_from_file
from modeldiff.core.parser import Parser


def test(args: argparse.ArgumentParser):
    xadd_model_1, context = get_xadd_model_from_file(args.f_env1, args.f_inst1)
    xadd_model_2, _ = get_xadd_model_from_file(args.f_env2, args.f_inst2, context=context)

    parser = Parser()
    mdp = parser.parse(xadd_model_1)

    pe = PolicyEvaluation(mdp, args.iter)
    pe.solve()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--f_env1", type=str, default="RDDL/Navigation_disc_goal/domain.rddl")
    parser.add_argument("--f_inst1", type=str, default=None)
    parser.add_argument("--f_env2", type=str, default="RDDL/Navigation_disc_goal2/domain.rddl")
    parser.add_argument("--f_inst2", type=str, default=None)
    parser.add_argument("--iter", type=int, default=2)

    test()
