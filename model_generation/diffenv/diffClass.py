from pySDP.utils.utils import get_xadd_model_from_file
from pySDP.core.parser import Parser
from pySDP.core.mdp import MDP
from pySDP.core.policy import Policy
from pySDP.core.action import Action
from pySDP.policy_evaluation.pe import PolicyEvaluation
from pySDP.value_iteration.vi import ValueIteration

from xaddpy.xadd.xadd import XADD

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

import json

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

    def load_policy(self, mdp: MDP, context: XADD) -> Policy:
        policy_dict_str = json.load(open(self._policy_path, 'r'))
        xadd_policy = {}
        for aname, action in mdp.actions.items():
            policy_id = context.import_xadd(xadd_str=policy_dict_str[aname])
            xadd_policy[aname] = policy_id
        policy = Policy(mdp)
        policy_dict = {}
        for aname, action in mdp.actions.items():
            policy_dict[action] = xadd_policy[aname]
        policy.load_policy(policy_dict)
        return policy

    
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
        
        policy_move_east = """
                        ( [pos_x___a1 - 5 <= 0] 
                            ( [0]
                            )
                            ( [pos_x___a1 - 8 <= 0]
                                ( [1] )
                                ( [0] )
                            )
                        )
                    """

        policy_move_west = """
                                ( [pos_x___a1 - 5 <= 0] 
                                    ( [pos_x___a1 - 2 <= 0]
                                        ( [0] )
                                        ( [1] )
                                    )
                                    ( [0]
                                    )
                                )
                            """

        policy_move_north = """
                                ( [pos_y___a1 - 8 <= 0] 
                                    ( [1] ) 
                                    ( [0] )
                                )
                                """

        policy_move_south = """
                                ( [pos_y___a1 - 10 <= 0] 
                                    ( [0] )
                                    ( [1] )
                                )
                                """

        policy_move_east = context.import_xadd(xadd_str=policy_move_east, locals=mdp.model.ns)
        policy_move_east = context.reduce_lp(policy_move_east)
        policy_move_east_false = context.apply(context.ONE, policy_move_east, 'subtract')

        policy_move_west = context.import_xadd(xadd_str=policy_move_west, locals=mdp.model.ns)
        policy_move_west = context.reduce_lp(policy_move_west)
        policy_move_west =  context.apply(policy_move_west, policy_move_east_false, 'prod')

        policy_move_east_west = context.apply(policy_move_east, policy_move_west, 'max')
        policy_move_east_west_false = context.apply(context.ONE, policy_move_east_west, 'subtract')

        policy_move_north = context.import_xadd(xadd_str=policy_move_north, locals=mdp.model.ns)
        policy_move_north = context.reduce_lp(policy_move_north)
        policy_move_north = context.apply(policy_move_north, policy_move_east_west_false, 'prod')

        policy_move_east_west_north = context.apply(policy_move_east_west, policy_move_north, 'max')
        policy_move_east_west_north_false = context.apply(context.ONE, policy_move_east_west_north, 'subtract')


        policy_move_south = context.import_xadd(xadd_str=policy_move_south, locals=mdp.model.ns)
        policy_move_south = context.reduce_lp(policy_move_south)
        policy_move_south = context.apply(policy_move_south, policy_move_east_west_north_false, 'prod')

        policy_move_east_west_north_south = context.apply(policy_move_east_west_north, policy_move_south, 'max')
        policy_move_east_west_north_south_false = context.apply(context.ONE, policy_move_east_west_north_south, 'subtract')

        do_nothing = context.apply(context.ONE, policy_move_east_west_north_south, 'subtract')
        do_nothing = context.reduce_lp(do_nothing)

        # make a dictionary of action as string to node id
        xadd_policy = {
            'move_east___a1': policy_move_east,
            'move_west___a1': policy_move_west,
            'move_north___a1': policy_move_north,
            'move_south___a1': policy_move_south,
            'do_nothing___a1' : do_nothing
            }

        policy_dict = {}
        policy = Policy(mdp)

        for aname, action in mdp.actions.items():
            policy_dict[action] = xadd_policy[aname]
        policy.load_policy(policy_dict)
        return policy
    
    def create_policy_wildfire(self, mdp: MDP, context: XADD) -> Policy:
        xadd_policy = {}
        # all_id = []
        for aname, action in mdp.actions.items():
            policy_id = context.ONE
            for s, b in action._bool_dict.items():
                if b == True:
                    policy_str = "( [{}] ( [1] ) ( [0] ) )".format(s)
                else:
                    policy_str = "( [{}] ( [0] ) ( [1] ) )".format(s)
                a_id = context.import_xadd(xadd_str=policy_str)
                policy_id = context.apply(policy_id, a_id, 'prod')
            xadd_policy[aname] = policy_id 

            # print(aname)
            # print(context._id_to_node[policy_id])
            # all_id.append(policy_id)

        policy = Policy(mdp)
        policy_dict = {}

        for aname, action in mdp.actions.items():
            policy_dict[action] = xadd_policy[aname]
        policy.load_policy(policy_dict)
        
        ## Debugging to see if all actions add to 1
        # sum_id = context.ZERO
        # n = 0
        # for a in all_id:
        #     sum_id = context.apply(sum_id, a, 'add')
        #     n += 1
        #     print(n)
        #     print(context._id_to_node[sum_id])
        # print(context._id_to_node[sum_id])

        return policy
        
    

    
    def do_SDP(self, model, context, mode="PE", discount=0.9, t=2):
        parser = Parser()
        mdp = parser.parse(model, is_linear=True, discount=discount)
        if "reservoir" in self._domain_type:
            policy = self.create_policy_reservoir(mdp, context)
        elif "inventory" in self._domain_type:
            policy = self.create_policy_inventory(mdp, context)
        elif "navigation" in self._domain_type:
            # policy = self.create_policy_navigation(mdp, context)
            policy = self.load_policy(mdp, context)
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

    

