from modeldiff.utils.utils import get_xadd_model_from_file
from modeldiff.core.parser import Parser
from modeldiff.core.mdp import MDP
from modeldiff.core.policy import Policy
from modeldiff.core.action import Action
from modeldiff.policy_evaluation.pe import PolicyEvaluation
from modeldiff.value_iteration.vi import ValueIteration

from xaddpy.xadd.xadd import XADD

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


import os

class ModelDiffReservoir:
    def __init__(self, 
                model_1_path:str, model_2_path:str,
                policy_path:str=None,
                inst_1_path:str=None,
                inst_2_path:str=None):
        self._policy_path = policy_path
        self._model_1_path = model_1_path
        self._inst_1_path = inst_1_path
        self._model_2_path = model_2_path
        self._inst_2_path = inst_2_path
        self._model_1, self._context_1 = get_xadd_model_from_file(model_1_path, inst_1_path)
        self._model_2, self._context_2 = get_xadd_model_from_file(model_2_path, inst_2_path)
        self._model_diff = None
        self._context_diff = None
        self._pe_dict = {}
    
    
    def build_model_with_diff_reward(self):
        self._model_diff, self._context_diff = get_xadd_model_from_file(self._model_2_path, self._inst_2_path)
        r1_path = self._context_1.export_xadd(self._model_1.reward, 'temp1.xadd')
        r1_node = self._context_diff.import_xadd(fname='temp1.xadd', locals=self._context_1._str_var_to_var)
        os.remove('temp1.xadd')
        diff_node = self._context_diff.apply(self._model_diff.reward, r1_node, 'subtract')
        diff_node = self._context_diff.reduce_lp(diff_node)
        self._model_diff.reward = diff_node
        
        return diff_node
    
    def create_policy(self, mdp: MDP, context: XADD) -> Policy:
        release = context.import_xadd(self._policy_path + 'release_t.xadd', locals=context._str_var_to_var)
        do_nothing = context.import_xadd(self._policy_path + 'do_nothing.xadd', locals=context._str_var_to_var)
        xadd_policy = {
            'release___t1': release,
            'do_nothing___t1': do_nothing
        }
        policy = Policy(mdp)
        policy_dict = {}
        for aname, action in mdp.actions.items():
            policy_dict[action] = xadd_policy[aname]
        policy.load_policy(policy_dict)
        return policy
    
    def do_PE(self, model, context, t=2):
        parser = Parser()
        mdp = parser.parse(model, is_linear=True)
        policy = self.create_policy(mdp, context)


        pe = PolicyEvaluation(mdp, policy, t)
        iter_id = pe.solve()

        # print(pe.context._id_to_node.get(model.reward))
        # print(iter_id)
        # print(pe.context._id_to_node.get(iter_id))

        return iter_id
    
    def do_VI(self, model, context, t=2):
        parser = Parser()
        mdp = parser.parse(model, is_linear=True)
        policy = self.create_policy(mdp, context)
        pe = ValueIteration(mdp, policy, t)
        iter_id = pe.solve()
        return iter_id
    
    def eval_function(self, x, y, iter_id, model, context):
        pos_x = model.ns['pos-x___a1']
        pos_y = model.ns['pos-y___a1']
        b_assign = {}
        c_assign = {pos_x:x, pos_y:y}
        res = context.evaluate(iter_id, bool_assign=b_assign, cont_assign=c_assign)
        return res


    def gen_countour_graph(self, X, Y, Z, title, fpath):
        fig = plt.figure(figsize=(6,5))
        left, bottom, width, height = 0.1, 0.1, 0.8, 0.8
        ax = fig.add_axes([left, bottom, width, height])

        cp = plt.contourf(X, Y, Z, np.arange(-2.5, 2.5, .1), cmap='RdGy')
        plt.colorbar(cp)

        ax.set_title(title)
        ax.set_xlabel('x')
        ax.set_ylabel('y')

        plt.xticks(range(0,11))
        plt.yticks(range(0,11))

        plt.savefig(fpath) 
        plt.cla()

    
    def gen_value_graph(self, title, fpath, model, context, pe_id):
        x = np.linspace(0,10,50)
        y = np.linspace(0,10,50)
        X, Y = np.meshgrid(x,y)
        Z = np.zeros((len(x), len(y)))
        for i in range(len(x)):
            for j in range(len(y)):
                Z[i][j] = self.eval_function(X[i][j], Y[i][j], pe_id, model, context)
        Z += 0.0000001
        self.gen_countour_graph(X, Y, Z, title, fpath)
    
    def gen_XADD_graph(self, node_id, fpath):
        return


        


    




       





# m1_path = "../RDDL/Navigation_disc_goal_551010/domain.rddl"
# m2_path = "../RDDL/Navigation_disc_goal_771010/domain.rddl"

# model_diff = ModelDiff(m1_path, m2_path)
# model_diff.build_model_with_diff_reward()

    

