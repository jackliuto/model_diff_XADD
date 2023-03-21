from pyRDDLGym import RDDLEnv
from pyRDDLGym import ExampleManager
from pyRDDLGym.Policies.Agents import RandomAgent
from pyRDDLGym.Core.Grounder.RDDLGrounder import RDDLGrounder
from pyRDDLGym.XADD.RDDLModelXADD import RDDLModelWXADD
from pyRDDLGym.Core.Parser.RDDLReader import RDDLReader
from pyRDDLGym.Core.Parser.parser import RDDLParser

from collections import OrderedDict

import sympy
from xaddpy import XADD

from xaddpy.xadd.xadd_parse_utils import parse_xadd_grammar


# gen xadd model from a RDDLEnv
def gen_xadd_model(rddlModelLifted):
    grounder = RDDLGrounder(rddlModelLifted._AST)
    grounded_model = grounder.Ground()
    xadd_model = RDDLModelWXADD(grounded_model)
    xadd_model.compile()
    return xadd_model, grounded_model



def gen_action(possible_actions, state):
    if state['pos-x___a1'] < 10:
        return possible_actions[0]
    elif state['pos-y___a1'] < 10:
        return possible_actions[1]
    else:
        return possible_actions[2]


# def xadd_str2list(xadd_str):
#     xadd_str = xadd_str.replace('\n', '')
#     if xadd_str.rfind('(') == 0 and xadd_str.rfind('[') == 2:
#         xadd_as_list = [sympy.sympify(xadd_str.strip('( [] )'))]
#     else: 
#         xadd_as_list = parse_xadd_grammar(xadd_str, ns={})[1][0]
#     return xadd_as_list
        


def add_model_reward(xadd_model_1, xadd_model_2):
    context_new = XADD()

    context_1 = xadd_model_1._context
    context_2 = xadd_model_2._context
    # reward_1 = context_1._id_to_node.get(xadd_model_1.reward)
    # reward_2 = context_2._id_to_node.get(xadd_model_2.reward)

    context_1.export_xadd(xadd_model_1.reward, 'temp_node_1.xadd')
    context_2.export_xadd(xadd_model_2.reward, 'temp_node_2.xadd')

    node_id_1 = context_new.import_xadd(fname='temp_node_1.xadd')
    node_id_2 = context_new.import_xadd(fname='temp_node_2.xadd')



    # reward_1.turn_off_print_node_info()
    # reward_2.turn_off_print_node_info()
    
    # reward_1_lst = xadd_str2list(str(reward_1))
    # reward_2_lst = xadd_str2list(str(reward_2))

    # print(reward_1_lst)
    # print(reward_2_lst)

    # node_0 = context_new.build_initial_xadd(reward_1_lst)
    # node_1 = context_new.build_initial_xadd(reward_2_lst)

    node_diff = context_new.apply(node_id_1, node_id_2, 'add')



    print(context_new._id_to_node.get(node_id_1))
    print(context_new._id_to_node.get(node_id_2))
    print(context_new._id_to_node.get(node_diff))

    return context_new







def test():

    domain_path = '../RDDL/Navigation_disc_goal/domain.rddl'
    instance_path = '../RDDL/Navigation_disc_goal/instance0.rddl'

    possible_actions = [OrderedDict([
                        ('move-pos-x___a1', 1), 
                        ('move-pos-y___a1', 0), 
                        ('do-nothing___a1', 0)
                        ]),
                        OrderedDict([
                        ('move-pos-x___a1', 0), 
                        ('move-pos-y___a1', 1), 
                        ('do-nothing___a1', 0)
                        ]),
                        OrderedDict([
                        ('move-pos-x___a1', 0), 
                        ('move-pos-y___a1', 0), 
                        ('do-nothing___a1', 1)
                        ]),]


    myEnv = RDDLEnv.RDDLEnv(domain=domain_path, instance=instance_path)
    xadd_model_init, grounded_model_init =  gen_xadd_model(myEnv.model)

    state = myEnv.reset()
    action = gen_action(possible_actions, state)
    next_state, reward, done, info = myEnv.step(action)


    xadd_model_1, grounded_model_1 = gen_xadd_model(myEnv.model)
    new_context = add_model_reward(xadd_model_init, xadd_model_1)






test()




# myEnv.step(posible_actions[0])
# xadd_model = gen_xadd_model(myEnv)




