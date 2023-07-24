from SDP.utils.utils import get_xadd_model_from_file
from SDP.core.parser import Parser
from SDP.core.mdp import MDP
from SDP.core.policy import Policy
from SDP.core.action import Action
from SDP.policy_evaluation.pe import PolicyEvaluation
from SDP.value_iteration.vi import ValueIteration

from xaddpy.xadd.xadd import XADD

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


import os

class ModelDiff:
    def __init__(self, 
                domain_type:str, 
                model_1_path:str, 
                model_2_path:str,
                policy_path:str=None,
                inst_1_path:str=None,
                inst_2_path:str=None):
        self.domain_type = domain_type
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
    
    def create_policy_reservoir(self, mdp: MDP, context: XADD) -> Policy:
        release_t1_true = context.import_xadd(self._policy_path + 'release_t1_true.xadd', locals=context._str_var_to_var)
        release_t1_false = context.import_xadd(self._policy_path + 'release_t1_false.xadd', locals=context._str_var_to_var)

        release_t1_true_t2_true = context.import_xadd(self._policy_path + 'release_t1_true_t2_true.xadd', locals=context._str_var_to_var)
        release_t1_true_t2_false = context.import_xadd(self._policy_path + 'release_t1_true_t2_false.xadd', locals=context._str_var_to_var)
        release_t1_false_t2_true = context.import_xadd(self._policy_path + 'release_t1_false_t2_true.xadd', locals=context._str_var_to_var)
        release_t1_false_t2_false = context.import_xadd(self._policy_path + 'release_t1_false_t2_false.xadd', locals=context._str_var_to_var)
        xadd_policy = {
            "{release___t1: True}": release_t1_true,
            "{release___t1: False}": release_t1_false,
            "{release___t1: True, release___t2: True}": release_t1_true_t2_true,
            "{release___t1: True, release___t2: False}": release_t1_true_t2_false,
            "{release___t1: False, release___t2: True}": release_t1_false_t2_true,
            "{release___t1: False, release___t2: False}": release_t1_false_t2_false,
        }

        policy = Policy(mdp)
        policy_dict = {}

        for aname, action in mdp.actions.items():
            policy_dict[action] = xadd_policy[aname]
        policy.load_policy(policy_dict)
        return policy
    
    def do_PE(self, model, context, discount=0.9, t=2):
        parser = Parser()
        mdp = parser.parse(model, is_linear=True, discount=discount)
        if self.domain_type == "reservoir":
            policy = self.create_policy_reservoir(mdp, context)
        else:
            raise ValueError("{} not implemneted".foramt(self.domain_type))


        pe = PolicyEvaluation(mdp, policy, t)
        iter_id, q_list = pe.solve()

        # print(pe.context._id_to_node.get(model.reward))
        # print(iter_id)
        # print(pe.context._id_to_node.get(iter_id))

        return iter_id, q_list
    
    def do_VI(self, model, context, t=2):
        parser = Parser()
        mdp = parser.parse(model, is_linear=True)
        policy = self.create_policy(mdp, context)
        pe = ValueIteration(mdp, policy, t)
        iter_id = pe.solve()
        return iter_id
    
    def eval_function(self, b_dict, c_dict, iter_id, model, context):
        b_assign = {}
        c_assign = {}
        # pos_x = model.ns['pos-x___a1']
        # pos_y = model.ns['pos-y___a1']
        for k,v in b_dict.items():
            b_assign[model.ns[k]] = v
        for k,v in c_dict.items():
            c_assign[model.ns[k]] = v
        
        # b_assign = {}
        # c_assign = {pos_x:x, pos_y:y}
        res = context.evaluate(iter_id, bool_assign=b_assign, cont_assign=c_assign)
        return res

    

        


    




       





# m1_path = "../RDDL/Navigation_disc_goal_551010/domain.rddl"
# m2_path = "../RDDL/Navigation_disc_goal_771010/domain.rddl"

# model_diff = ModelDiff(m1_path, m2_path)
# model_diff.build_model_with_diff_reward()

    

