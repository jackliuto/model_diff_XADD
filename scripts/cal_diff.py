# This Script calculate the differences between two RDDL Model according to their cpfs in XADD
# THe script will save the difference into two image folders

from pathlib import Path
from typing import Optional
from collections import defaultdict

from pyRDDLGym.Core.Grounder.RDDLGrounder import RDDLGrounder
from pyRDDLGym.XADD.RDDLModelXADD import RDDLModelWXADD
from pyRDDLGym.Core.Parser.RDDLReader import RDDLReader
from pyRDDLGym.Core.Parser.parser import RDDLParser
from pyRDDLGym.Examples.ExampleManager import ExampleManager

from xaddpy.xadd.xadd_parse_utils import parse_xadd_grammar
from xaddpy.xadd import XADD
import sympy
from pprint import pprint

from file_path import *

def parse_xadd_model(domain_name, domain_path, instance_path):

    reader = RDDLReader(domain_path, instance_path)
    domain = reader.rddltxt
    parser = RDDLParser(None, False)
    parser.build()

    rddl_ast = parser.parse(domain)
    grounder = RDDLGrounder(rddl_ast)

    model = grounder.Ground()

    xadd_model = RDDLModelWXADD(model)

    return xadd_model

def xadd_str2list(xadd_str):
    xadd_str = xadd_str.replace('\n', '')
    if xadd_str.rfind('(') == 0 and xadd_str.rfind('[') == 2:
        xadd_as_list = [sympy.sympify(xadd_str.strip('( [] )'))]
    else: 
        print(xadd_str)
        xadd_as_list = parse_xadd_grammar(xadd_str, ns={})[1][0]
    return xadd_as_list
        
    
def xadd_caldiff(xadd_list0, xadd_list1):
    new_context = XADD()
    node_id0 = new_context.build_initial_xadd(xadd_list0)
    node_id1 = new_context.build_initial_xadd(xadd_list1)
    # caulate node_1 - node_0
    node_diff_id = new_context.apply(node_id1, node_id0, 'subtract')
    node_diff = new_context._id_to_node.get(node_diff_id)
    node_diff.turn_off_print_node_info()
    node_diff_list = xadd_str2list(str(node_diff))
    return node_diff_list
    

def save_xadd_graph(xadd_list,f_path):
    new_context = XADD()
    node = new_context.build_initial_xadd(xadd_list)
    new_context.save_graph(node, f_path)
    return f_path


def test_model_diff():

    domain_name = DOMAIN_NAME
    domain_path = DOMAIN_PATH
    instance_path = INSTANCE_PATH

    domain_name_new = DOMAIN_NAME_NEW
    domain_path_new = DOMAIN_PATH_NEW
    instance_path_new = INSTANCE_PATH_NEW

    xadd_model0 = parse_xadd_model(domain_name_new, domain_path_new, instance_path_new)
    xadd_model1 = parse_xadd_model(domain_name, domain_path, instance_path)

    xadd_model0.compile()
    context0 = xadd_model0._context

    xadd_model1.compile()
    context1 = xadd_model1._context

    xadd_diff_dict = defaultdict(list)

    for cpf, node_id in xadd_model0.cpfs.items():
        save_node = context0._id_to_node.get(node_id)
        save_node.turn_off_print_node_info()
        xadd_diff_dict[cpf].append(xadd_str2list(str(save_node)))

    for cpf, node_id in xadd_model1.cpfs.items():
        save_node = context1._id_to_node.get(node_id)
        save_node.turn_off_print_node_info()
        xadd_diff_dict[cpf].append(xadd_str2list(str(save_node)))

    reward_node0 = context0._id_to_node.get(xadd_model0.reward)
    reward_node0.turn_off_print_node_info()

    xadd_diff_dict['reward'].append(xadd_str2list(str(reward_node0)))

    reward_node1 = context1._id_to_node.get(xadd_model1.reward)
    reward_node1.turn_off_print_node_info()
    xadd_diff_dict['reward'].append(xadd_str2list(str(reward_node1)))


    for k,v in xadd_diff_dict.items():
        xadd_diff_dict[k].append(xadd_caldiff(*v))
        

    # Path(f'tmp/{domain_name_new}').mkdir(parents=True, exist_ok=True)
    # f_path = f"{domain_name_new}/{domain_name_new}_reward_diff"

    # print(f_path)
    # save_xadd_graph(xadd_diff_dict['reward'][2], f_path)
    
    return xadd_diff_dict


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