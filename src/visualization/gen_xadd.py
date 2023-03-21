from pathlib import Path
from typing import Optional
from pyRDDLGym.Core.Grounder.RDDLGrounder import RDDLGrounder
from pyRDDLGym.XADD.RDDLModelXADD import RDDLModelWXADD
from pyRDDLGym.Core.Parser.RDDLReader import RDDLReader
from pyRDDLGym.Core.Parser.parser import RDDLParser
from pyRDDLGym.Examples.ExampleManager import ExampleManager


def gen_diff(xadd_model0, xadd_model1):
    xadd_model0.compile()
    context0 = xadd_model0._context

    xadd_model1.compile()
    context1 = xadd_model1._context
    
    cpfs_diff_dict = {}
    for cpf, expr in xadd_model0.cpfs.items():
        if cpf not in xadd_model1.cpfs.keys():
            cpfs_diff_dict[cpf] = expr
        else:
            node0 = context0.get_exist_node(expr)
            node1 = context1.get_exist_node(xadd_model1.cpfs[cpf])
            if not node_identical(node0, node1):
                cpfs_diff_dict[cpf] = expr
    
    reward_node0 = context0.get_exist_node(xadd_model0.reward)
    reward_node1 = context1.get_exist_node(xadd_model1.reward)

    identical_reward = False

    if node_identical(reward_node0, reward_node1):
        identical_reward = True

    return cpfs_diff_dict, identical_reward


def node_identical(node0, node1):
    if node0._is_leaf and node1._is_leaf:
        if node0 == node1:
            return True

    if not node0._is_leaf and not node1._is_leaf:
        node0_high = node0._context.get_exist_node(node0._high)
        node0_low = node0._context.get_exist_node(node0._low)

        node1_high = node1._context.get_exist_node(node1._high)
        node1_low = node1._context.get_exist_node(node1._low)

        return (node0 == node1) and (node_identical(node0_high, node1_high)) and (node_identical(node0_low, node1_low))
    
    return False


def parse_xadd_model(domain_name, domain_path, instance_path):
    # env_info = ExampleManager.GetEnvInfo(domain_name)
    # domain = env_info.get_domain()
    # instance = env_info.get_instance(0)

    reader = RDDLReader(domain_path, instance_path)
    domain = reader.rddltxt
    parser = RDDLParser(None, False)
    parser.build()

    rddl_ast = parser.parse(domain)
    grounder = RDDLGrounder(rddl_ast)

    model = grounder.Ground()

    xadd_model = RDDLModelWXADD(model)

    return xadd_model


def save_all_graph(xadd_model, domain_name):
    Path(f'tmp/{domain_name}').mkdir(parents=True, exist_ok=True)
        
    for cpf_, expr in xadd_model.cpfs.items():
        cpf = cpf_.strip("'")
        f_path = f"{domain_name}/{domain_name}_inst0_{cpf}"
        xadd_model._context.save_graph(expr, f_path)
    f_path = f"{domain_name}/{domain_name}_inst0_reward"
    xadd_model._context.save_graph(xadd_model.reward, f_path)



def test_model_diff():

    # domain_name = 'PowerGen_linear'
    # domain_path = '../RDDL/PowerGen_linear/domain.rddl'
    # instance_path = '../RDDL/PowerGen_linear/instance0.rddl'

    # domain_name_new = 'PowerGen_linear_higherDemandPen'
    # domain_path_new = '../RDDL/PowerGen_linear_higherDemandPen/domain.rddl'
    # instance_path_new = '../RDDL/PowerGen_linear_higherDemandPen/instance0.rddl'

    # domain_name = 'Navigation_cont'
    # domain_path = '../RDDL/Navigation_cont/domain.rddl'
    # instance_path = '../RDDL/Navigation_cont/instance0.rddl'

    # domain_name_new = 'Navigation_cont_hole'
    # domain_path_new = '../RDDL/Navigation_cont_hole/domain.rddl'
    # instance_path_new = '../RDDL/Navigation_cont_hole/instance0.rddl'

    # domain_name = 'Navigation_cont'
    # domain_path = '../RDDL/Navigation_cont/domain.rddl'
    # instance_path = '../RDDL/Navigation_cont/instance0.rddl'

    # domain_name_new = 'Navigation_cont_border'
    # domain_path_new = '../RDDL/Navigation_cont_border/domain.rddl'
    # instance_path_new = '../RDDL/Navigation_cont_border/instance0.rddl'

    # domain_name = 'Navigation_disc'
    # domain_path = '../RDDL/Navigation_disc/domain.rddl'
    # instance_path = '../RDDL/Navigation_disc/instance0.rddl'

    # domain_name_new = 'Navigation_disc2'
    # domain_path_new = '../RDDL/Navigation_disc2/domain.rddl'
    # instance_path_new = '../RDDL/Navigation_disc2/instance0.rddl'

    # domain_name = 'Navigation_disc_linear'
    # domain_path = '../RDDL/Navigation_disc_linear/domain.rddl'
    # instance_path = '../RDDL/Navigation_disc/instance0.rddl'

    # domain_name_new = 'Navigation_disc_linear2'
    # domain_path_new = '../RDDL/Navigation_disc_linear2/domain.rddl'
    # instance_path_new = '../RDDL/Navigation_disc_linear2/instance0.rddl'

    # domain_name = 'Navigation_disc_linear'
    # domain_path = '../RDDL/Navigation_disc_linear/domain.rddl'
    # instance_path = '../RDDL/Navigation_disc/instance0.rddl'

    # domain_name_new = 'Navigation_disc_linear_policy'
    # domain_path_new = '../RDDL/Navigation_disc_linear_policy/domain.rddl'
    # instance_path_new = '../RDDL/Navigation_disc_linear_policy/instance0.rddl'


    # domain_name = 'Navigation_cont'
    # domain_path = '../RDDL/Navigation_cont/domain.rddl'
    # instance_path = '../RDDL/Navigation_cont/instance0.rddl'

    # domain_name_new = 'Navigation_cont_error'
    # domain_path_new = '../RDDL/Navigation_cont_error/domain.rddl'
    # instance_path_new = '../RDDL/Navigation_cont_error/instance0.rddl'

    domain_name = 'Navigation_disc_goal'
    domain_path = '../RDDL/Navigation_disc_goal/domain.rddl'
    instance_path = '../RDDL/Navigation_disc_goal/instance0.rddl'

    domain_name_new = 'Navigation_disc_goal2'
    domain_path_new = '../RDDL/Navigation_disc_goal2/domain.rddl'
    instance_path_new = '../RDDL/Navigation_disc_goal2/instance0.rddl'




    xadd_model0 = parse_xadd_model(domain_name_new, domain_path_new, instance_path_new)
    xadd_model1 = parse_xadd_model(domain_name, domain_path, instance_path)

    cpfs_diff_dict, identical_reward = gen_diff(xadd_model0, xadd_model1)

    print(cpfs_diff_dict, identical_reward)

    Path(f'tmp/{domain_name_new}').mkdir(parents=True, exist_ok=True)

    for cpf_, expr in cpfs_diff_dict.items():
        xadd_model0.print(expr)
        cpf = cpf_.strip("'")
        f_path = f"{domain_name_new}/{domain_name_new}_inst0_{cpf}"
        xadd_model0._context.save_graph(expr, f_path)
    if not identical_reward:
        f_path = f"{domain_name_new}/{domain_name_new}_inst0_reward"
        xadd_model0._context.save_graph(xadd_model0.reward, f_path)

    save_all_graph(xadd_model0, domain_name_new)
    save_all_graph(xadd_model1, domain_name)



if __name__ == "__main__":

    test_model_diff()

    # import argparse

    # parser = argparse.ArgumentParser()

    # parser.add_argument('--env', type=str, default='PowerGen discrete0',
    #                     help='The name of the RDDL environment')
    # parser.add_argument('--cpf', type=str, default=None,
    #                     help='If specified, only print out this CPF')
    # parser.add_argument('--save_graph', action='store_true',
    #                     help='Save the graph as pdf file', default=True)
    # args = parser.parse_args()
    # test_xadd(env_name=args.env, cpf=args.cpf, save_graph=args.save_graph)
