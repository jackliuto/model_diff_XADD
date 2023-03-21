import argparse

from modeldiff.policy_evaluation.pe import PolicyEvaluation
from modeldiff.utils.utils import get_xadd_model_from_file
from modeldiff.core.parser import Parser


def test(args: argparse.ArgumentParser):
    xadd_model_1, context = get_xadd_model_from_file(args.env_name1, args.instance1)
    xadd_model_2, _ = get_xadd_model_from_file(args.env_name2, args.instance2, context=context)

    parser = Parser()
    mdp = parser.parse(xadd_model_1)

    pe = PolicyEvaluation(mdp, args.iter)
    pe.solve()

if __name__ == "__main__":
    pass
