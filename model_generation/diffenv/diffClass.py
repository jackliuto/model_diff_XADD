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
                domain_path:str, 
                instance_source_path:str,
                instance_target_path:str,
                # model_1_path:str, 
                # model_2_path:str,
                policy_path:str=None):
        self._domain_type = domain_type
        self._policy_path = policy_path
        self._domain_path = domain_path
        # self._model_1_path = model_1_path
        # self._inst_1_path = inst_1_path
        # self._model_2_path = model_2_path
        # self._inst_2_path = inst_2_path
        self._instance_source_path = instance_source_path
        self._instance_target_path = instance_target_path
        self._model_1, self._context_1 = get_xadd_model_from_file(domain_path, instance_source_path)
        self._model_2, self._context_2 = get_xadd_model_from_file(domain_path, instance_target_path)
        self._model_diff = None
        self._context_diff = None
        self._pe_dict = {}
        self.THRESHOLD = 45
    
    
    def build_model_with_diff_reward(self):
        self._model_diff, self._context_diff = get_xadd_model_from_file(self._domain_path, self._instance_target_path)


        r1_path = self._context_1.export_xadd(self._model_1.reward, 'temp1.xadd')
        r1_node = self._context_diff.import_xadd(fname='temp1.xadd', locals=self._context_1._str_var_to_var)
        os.remove('temp1.xadd')
        diff_node = self._context_diff.apply(self._model_diff.reward, r1_node, 'subtract')
        diff_node = self._context_diff.reduce_lp(diff_node)
        self._model_diff.reward = diff_node
        
        return diff_node
    
    def create_policy_reservoir(self, mdp: MDP, context: XADD) -> Policy:
        threshold = self.THRESHOLD
        threshold = 45
        xadd_policy = {}
        for aname, action in mdp.actions.items():
            policy_id = context.ONE
            for i in aname[1:-1].split(','):
                res_name = i.strip().split('___')[1][0:2]
                bool_val = i.strip().split(' ')[1]
                if bool_val == "True":
                    policy_str = "( [rlevel___{} - {} <= 0] ( [0] ) ( [1] ) )".format(res_name, threshold)
                else:
                    policy_str = "( [rlevel___{} - {} <= 0] ( [1] ) ( [0] ) )".format(res_name, threshold)
                a_id = context.import_xadd(xadd_str=policy_str)
                policy_id = context.apply(policy_id, a_id, 'prod')
            xadd_policy[aname] = policy_id

        policy = Policy(mdp)
        policy_dict = {}

        for aname, action in mdp.actions.items():
            policy_dict[action] = xadd_policy[aname]
        policy.load_policy(policy_dict)
        return policy
    
    def create_policy_inventory(self, mdp: MDP, context: XADD) -> Policy:
        threshold = self.THRESHOLD
        threshold = 6
        xadd_policy = {}
        for aname, action in mdp.actions.items():
            policy_id = context.ONE
            for i in aname[1:-1].split(','):
                res_name = i.strip().split('___')[1][0:2]
                bool_val = i.strip().split(' ')[1]
                if bool_val == "True":
                    policy_str = "( [stock___{} - {} <= 0] ( [1] ) ( [0] ) )".format(res_name, threshold)
                else:
                    policy_str = "( [stock___{} - {} <= 0] ( [0] ) ( [1] ) )".format(res_name, threshold)
                a_id = context.import_xadd(xadd_str=policy_str)
                policy_id = context.apply(policy_id, a_id, 'prod')
            xadd_policy[aname] = policy_id

        policy = Policy(mdp)
        policy_dict = {}

        for aname, action in mdp.actions.items():
            policy_dict[action] = xadd_policy[aname]
        policy.load_policy(policy_dict)
        return policy
    
    def create_policy_navigation(self, mdp: MDP, context: XADD) -> Policy:
        threshold = self.THRESHOLD
        x_goal = 10
        y_goal = 10
        xadd_policy = {}
        for aname, action in mdp.actions.items():
            policy_id = context.ONE
            for i in aname[1:-1].split(','):
                agent_name = i.strip().split('___')[1][0:2]
                pos = i.strip().split('_')[2]
                bool_val = i.strip().split(' ')[1]
                if bool_val == "True":
                    if pos == 'x':
                        policy_str = "( [pos_{}___{} - {} <= 0] ( [1] ) ( [0] ) )".format(pos, agent_name, x_goal)
                    else:
                        policy_str = "( [pos_{}___{} - {} <= 0] ( [1] ) ( [0] ) )".format(pos, agent_name, y_goal)
                else:
                    if pos == 'x':
                        policy_str = "( [pos_{}___{} - {} <= 0] ( [0] ) ( [1] ) )".format(pos, agent_name, x_goal)
                    else:
                        policy_str = "( [pos_{}___{} - {} <= 0] ( [0] ) ( [1] ) )".format(pos, agent_name, y_goal)
                a_id = context.import_xadd(xadd_str=policy_str)
                policy_id = context.apply(policy_id, a_id, 'prod')
            xadd_policy[aname] = policy_id

        policy = Policy(mdp)
        policy_dict = {}

        for aname, action in mdp.actions.items():
            policy_dict[action] = xadd_policy[aname]
        policy.load_policy(policy_dict)
        return policy
    
    def create_policy_wildfire(self, mdp: MDP, context: XADD) -> Policy:
        xadd_policy = {}
        for aname, action in mdp.actions.items():
            print(aname)
            num_true = sum(action._bool_dict.values())
            if num_true == 0:
                pass
            elif num_true == 1:
                pass
            else:
                pass

            raise ValueError
            policy_id = context.ONE
            for i in aname[1:-1].split(','):
                agent_name = i.strip().split('___')[1][0:2]
                pos = i.strip().split('_')[2]
                bool_val = i.strip().split(' ')[1]
                if bool_val == "True":
                    if pos == 'x':
                        policy_str = "( [pos_{}___{} - {} <= 0] ( [1] ) ( [0] ) )".format(pos, agent_name, x_goal)
                    else:
                        policy_str = "( [pos_{}___{} - {} <= 0] ( [1] ) ( [0] ) )".format(pos, agent_name, y_goal)
                else:
                    if pos == 'x':
                        policy_str = "( [pos_{}___{} - {} <= 0] ( [0] ) ( [1] ) )".format(pos, agent_name, x_goal)
                    else:
                        policy_str = "( [pos_{}___{} - {} <= 0] ( [0] ) ( [1] ) )".format(pos, agent_name, y_goal)
                a_id = context.import_xadd(xadd_str=policy_str)
                policy_id = context.apply(policy_id, a_id, 'prod')
            xadd_policy[aname] = policy_id

        policy = Policy(mdp)
        policy_dict = {}

        for aname, action in mdp.actions.items():
            policy_dict[action] = xadd_policy[aname]
        policy.load_policy(policy_dict)
        return policy
    

    
    def do_SDP(self, model, context, mode="PE", discount=0.9, t=2):
        parser = Parser()
        mdp = parser.parse(model, is_linear=True, discount=discount)
        if "reservoir" in self._domain_type:
            policy = self.create_policy_reservoir(mdp, context)
        elif "inventory" in self._domain_type:
            policy = self.create_policy_inventory(mdp, context)
        elif "navigation" in self._domain_type:
            policy = self.create_policy_navigation(mdp, context)
        elif "wildfire" in self._domain_type:
            policy = self.create_policy_wildfire(mdp, context)
        else:
            raise ValueError("{} not implemneted".format(self._domain_type))

        if mode == "PE":
            pe = PolicyEvaluation(mdp, policy, t)
            iter_id, q_list = pe.solve()
        elif mode == "VI":
            vi = ValueIteration(mdp, t)
            iter_id, q_list = vi.solve()

        # print(pe.context._id_to_node.get(model.reward))
        # print(iter_id)
        # print(pe.context._id_to_node.get(iter_id))

        return iter_id, q_list
    
    def do_VI(self, model, context, t=2):
        parser = Parser()
        mdp = parser.parse(model, is_linear=True)
        policy = self.create_policy(mdp, context)
        pe = ValueIteration(mdp, t)
        iter_id = pe.solve()
        return iter_id
    
    def eval_function(self, b_dict, c_dict, iter_id, model, context):
        b_assign = {}
        c_assign = {}
        # print(b_dict)
        # print(c_dict)
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

    

