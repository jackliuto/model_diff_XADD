import argparse
from xaddpy.xadd.xadd import XADD

from modeldiff.core.mdp import MDP
from modeldiff.core.policy import Policy
from modeldiff.core.action import Action
from modeldiff.policy_evaluation.pe import PolicyEvaluation
from modeldiff.utils.utils import get_xadd_model_from_file
from modeldiff.core.parser import Parser

import numpy as np
import matplotlib.pyplot as plt


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


def do_PE(model, context):
    parser = Parser()
    mdp = parser.parse(model, is_linear=True)
    policy = create_policy(mdp, context)

    pe = PolicyEvaluation(mdp, policy, args.iter)
    iter_id = pe.solve()
    return iter_id


def save_xadd_graph(iter_id, context, fpath):
    # parser = Parser()
    # mdp = parser.parse(model, is_linear=True)
    # policy = create_policy(mdp, context)

    # pe = PolicyEvaluation(mdp, policy, args.iter)
    # iter_id = pe.solve()

    context.save_graph(iter_id, fpath)



def eval_function(x,y, iter_id, model, context):
    pos_x = model.ns['pos-x___a1']
    pos_y = model.ns['pos-y___a1']

    b_assign = {}
    c_assign = {pos_x:x, pos_y:y}

    res = context.evaluate(iter_id, bool_assign=b_assign, cont_assign=c_assign)

    return res




def gen_countour_graph(X, Y, Z, title, fname):
    fig = plt.figure(figsize=(6,5))
    left, bottom, width, height = 0.1, 0.1, 0.8, 0.8
    ax = fig.add_axes([left, bottom, width, height])

    cp = plt.contourf(X, Y, Z)
    plt.colorbar(cp)

    ax.set_title(title)
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    plt.xticks(range(0,11))
    plt.yticks(range(0,11))

    plt.savefig(fname) 
    plt.cla()

    return fname
    

    # Z = np.sin(X) + np.cos(Y)

    # print(np.sin(X))



def test(args: argparse.ArgumentParser):
    xadd_model_1, context1 = get_xadd_model_from_file(args.f_env1, args.f_inst1)
    xadd_model_2, context2 = get_xadd_model_from_file(args.f_env2, args.f_inst2)
    xadd_model_diff, context_diff = get_xadd_model_from_file(args.f_env1, args.f_inst1)


    r1_path = context1.export_xadd(xadd_model_1.reward, 'temp1.xadd')
    r1_node = context_diff.import_xadd(fname='temp1.xadd', locals=context1._str_var_to_var)
    r2_path = context2.export_xadd(xadd_model_2.reward, 'temp2.xadd')
    r2_node = context_diff.import_xadd(fname='temp2.xadd', locals=context2._str_var_to_var)

    diff_node = context_diff.apply(xadd_model_diff.reward, r2_node, 'subtract')
    context_diff.reduce_lp(diff_node)
    xadd_model_diff.reward = diff_node

    # context_diff.save_graph(xadd_model_diff.reward,'temp_graph_diff')

    iter_id_1 = do_PE(xadd_model_1, context1)
    iter_id_2 = do_PE(xadd_model_2, context2)
    iter_id_diff = do_PE(xadd_model_diff, context_diff)

    # gen_countour()

    # save_xadd_graph(iter_id_1, context1, 'v_1_t_2')
    # save_xadd_graph(iter_id_2, context2, 'v_2_t_2')
    # save_xadd_graph(iter_id_diff, context_diff, 'v_diff_t_2')

    x = np.linspace(0,10,50)
    y = np.linspace(0,10,50)

    X, Y = np.meshgrid(x,y)

    Z = np.zeros((len(x), len(y)))

    for i in range(len(x)):
        for j in range(len(y)):
            Z[i][j] = eval_function(X[i][j], Y[i][j], iter_id_1, xadd_model_1, context1)
    
    gen_countour_graph(X, Y, Z,'Value at T=2 for goal=(5,5)', 'visualization/vplot55.png')

    for i in range(len(x)):
        for j in range(len(y)):
            Z[i][j] = eval_function(X[i][j], Y[i][j], iter_id_2, xadd_model_2, context2)
    
    gen_countour_graph(X, Y, Z,'Value at T=2 for goal=(7,7)', 'visualization/vplot77.png')

    for i in range(len(x)):
        for j in range(len(y)):
            Z[i][j] = eval_function(X[i][j], Y[i][j], iter_id_diff, xadd_model_diff, context_diff)
    
    gen_countour_graph(X, Y, Z,'Value at T=2 for goal=(5,5) - goal=(7,7)', 'visualization/vplot_diff.png')




if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # parser.add_argument("--f_env1", type=str, default="RDDL/Navigation_disc_goal/domain.rddl")
    # parser.add_argument("--f_inst1", type=str, default=None)
    # parser.add_argument("--f_env2", type=str, default="RDDL/Navigation_disc_goal2/domain.rddl")
    # parser.add_argument("--f_inst2", type=str, default=None)

    parser.add_argument("--f_env1", type=str, default="RDDL/Navigation_disc_goal_551010/domain.rddl")
    parser.add_argument("--f_inst1", type=str, default=None)
    parser.add_argument("--f_env2", type=str, default="RDDL/Navigation_disc_goal_771010/domain.rddl")
    parser.add_argument("--f_inst2", type=str, default=None)


    parser.add_argument("--iter", type=int, default=2)
    args = parser.parse_args()
    test(args)