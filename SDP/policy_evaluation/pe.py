from typing import Set
import sympy as sp

from pyRDDLGym.XADD.RDDLModelXADD import RDDLModelWXADD
from xaddpy.xadd.xadd import XADD, DeltaFunctionSubstitution

from SDP.core.action import Action
from SDP.core.mdp import MDP
from SDP.core.policy import Policy
from SDP.utils.global_vars import SUM, PROD, RESTRICT_HIGH, RESTRICT_LOW, MAX

import pdb

class PolicyEvaluation:

    def __init__(self, mdp: MDP, policy: Policy, iter: int):
        """Performs policy evaluation for the specified number of iterations, or until convergence.
        
        Args:
            mdp (MDP): The MDP.
            policy (Policy): The policy to evaluate.
            iter (int): The number of iterations to perform.
        """
        self.mdp = mdp
        self.model = mdp.model
        self.policy = policy
        # self.policy_id = policy.compile_policy()
        self.policy_cpfs = self.gen_policy_cpfs()
        self._cur_iter = 0
        self._final_iter = 0
        self._n_iter = iter
        self._res_dd = 0
        self._prev_dd = 0
    

    def gen_policy_cpfs(self):
        policy_cpfs = {}
        # get transtion cpfs according to policy
        for cpf_name, cpf_id in self.model.cpfs.items():
            # collect all action vars in the cpfs
            action_vars = set([str(i) for i in self.context.collect_vars(cpf_id) if str(i) in self.mdp._actions.keys()])
            policy_cpf = cpf_id
            for a in action_vars:
                a_symbol = self.mdp._actions[a]._symbol

                # get state space where policy is True and False seperately
                a_policy_true = self.policy._dist[self.mdp._actions[a]]
                a_policy_false = self.context.apply(self.context.ONE, a_policy_true, 'subtract')

                # subsitute action with true and false in cpfs
                true_aciton_cpf = self.context.substitute(policy_cpf, {a_symbol:True})
                false_aciton_cpf = self.context.substitute(policy_cpf, {a_symbol:False})

                # find state space where transtion happens
                true_cpf = self.context.apply(true_aciton_cpf, a_policy_true, 'prod')
                false_cpf = self.context.apply(false_aciton_cpf, a_policy_false, 'prod')

                # marginalize all the true and false
                policy_cpf = self.context.apply(true_cpf, false_cpf, 'add')
                policy_cpf = self.mdp.standardize_dd(policy_cpf)
            policy_cpfs[cpf_name] = policy_cpf
        return policy_cpfs


    def solve(self) -> int:
        # Initialize the iteration counter
        self._cur_iter = 0

        # Initialize the value function to be the zero node
        value_dd = self.context.ZERO

        # Perform policy evaluation for the specified number of iterations, or until convergence
        while self._cur_iter < self._n_iter:

            self._prev_dd = value_dd
            
            # Compute the next value function
            value_dd = self.bellman_backup(value_dd)
            
            if self.mdp._is_linear:
                value_dd = self.mdp.standardize_dd(value_dd)

            # Increment the iteration counter
            self._cur_iter += 1
        self._final_iter = self._cur_iter
        return value_dd
    
    def bellman_backup(self, value_dd: int) -> int:
        """Performs a single iteration of the Bellman backup.
        
        Args:
            value_dd (int): The current value function.
        
        Returns:
            int: The next value function.
        """
        res_dd = self.context.ZERO      # Accumulate the value function in this variable

        # q_list = []
        
        # Iterate over all actions

        # Compute the action value function
        regr = self.regress(value_dd)

        # # get q value 
        # q_list.append((action._bool_dict, regr))

        # Multiply by pi(a|s)
        # Note: since everything's symbolic, state is not specified
        
        res_dd = self.context.apply(regr, res_dd, SUM)

        if self.mdp._is_linear:
            res_dd = self.mdp.standardize_dd(res_dd)
        
        return res_dd
    
    def regress(self, value_dd: int, regress_cont: bool = False) -> int:
        # Prime the value function
        subst_dict = self.mdp.prime_subs
        q = self.context.substitute(value_dd, subst_dict)

        # Discount
        if self.mdp.discount < 1.0:
            q = self.context.scalar_op(q, self.mdp.discount, PROD)

        # Add reward *if* it contains primed vars that need to be regressed
        i_and_ns_vars_in_reward = self.filter_i_and_ns_vars(self.context.collect_vars(self.model.reward))

        if len(i_and_ns_vars_in_reward) > 0:
            q = self.context.apply(q, self.model.reward, SUM)

        # # Get variables to eliminate
        # # TODO: Do we need to handle topological ordering?
        # # graph = self.mdp.build_dbn_dependency_dag(action, vars_to_regress)        
        vars_to_regress = self.filter_i_and_ns_vars(self.context.collect_vars(q), True, True)
        for v in vars_to_regress:
            if v in self.mdp._cont_ns_vars or v in self.mdp._cont_i_vars:
                q = self.regress_c_vars(q, v)
            elif v in self.mdp._bool_ns_vars or v in self.mdp._bool_i_vars:
                q = self.regress_b_vars(q, v)


        # i_vars = self.filter_i_vars(self.context.collect_vars(q), True, True)
        # i_vars = self.rank_vars(i_vars)
        
        # # Get variables to eliminate
        # # TODO: Do we need to handle topological ordering?
        # # graph = self.mdp.build_dbn_dependency_dag(action, vars_to_regress)        
        # vars_to_regress = self.filter_i_and_ns_vars(self.context.collect_vars(q), True, True)
        # while len(vars_to_regress) > 0:
        #     for v in vars_to_regress:
        #         if v in self.mdp._cont_ns_vars or v in self.mdp._cont_i_vars:
        #             q = self.regress_c_vars(q, v)
        #         elif v in self.mdp._bool_ns_vars or v in self.mdp._bool_i_vars:
        #             q = self.regress_b_vars(q, v)
        #     vars_to_regress = self.filter_i_and_ns_vars(self.context.collect_vars(q), True, True)
        



        # Add the reward
        if len(i_and_ns_vars_in_reward) == 0:
            q = self.context.apply(q, self.model.reward, SUM)

        # Continuous noise?
        # TODO
        # q = self.regress_noise(q, action)

        # # Continuous parameter
        # if regress_cont:
        #     q = self.regress_action(q, action)

        q = self.mdp.standardize_dd(q)

        return q

    def regress_c_vars(self, q: int, v: sp.Symbol) -> int:
        # Get the cpf for the variable
   
        cpf = self.policy_cpfs[str(v)]

        # Check regression cache
        key = (str(v), cpf, q)
        res = self.mdp._cont_regr_cache.get(key)

        if res is not None:
            return res

        # Perform regression via delta function substitution
        leaf_op = DeltaFunctionSubstitution(v, q, self.context)
        
        q = self.context.reduce_process_xadd_leaf(cpf, leaf_op, [], [])

        if self.mdp._is_linear:
            q = self.mdp.standardize_dd(q)
        
        # Cache result
        self.mdp._cont_regr_cache[key] = q


        return q
    
  
    def regress_b_vars(self, q: int, v: str) -> int:
        # Get the cpf for the variable
        
        cpf = self.policy_cpfs[str(v)]

        cpf_true = self.context.apply(cpf, self.context.ONE, 'add')
        cpf_true = self.context.apply(cpf_true, self.context.ONE, 'subtract')
        cpf_false = self.context.apply(self.context.ONE, cpf_true, 'subtract')

        dec_id = self.context._expr_to_id[self.mdp.model.ns[str(v)]]

        # # Marginalize out the variable
        # q = self.context.apply(q, cpf, PROD)
        restrict_high = self.context.op_out(q, dec_id, RESTRICT_HIGH)
        restrict_low = self.context.op_out(q, dec_id, RESTRICT_LOW)

        q_true = self.context.apply(cpf_true, restrict_high, 'prod')
        q_false = self.context.apply(cpf_false, restrict_low, 'prod')

        q = self.context.apply(q_true, q_false, SUM)

        return q
    


    
    def regress_action(self, q: int, a: Action) -> int:
        if len(a._action_params) == 0:  # No action parameters
            return q
        else:
            raise NotImplementedError("Continuous action parameters not yet supported")

    def filter_i_and_ns_vars(
            self, var_set: set, allow_bool: bool = True, allow_cont: bool = True
    ) -> Set[str]:
        filtered_vars = set()
        for v in var_set:
            if allow_cont and (v in self.mdp._cont_ns_vars or v in self.mdp._cont_i_vars):
                filtered_vars.add(v)
            elif allow_bool and (v in self.mdp._bool_ns_vars or v in self.mdp._bool_i_vars):
                filtered_vars.add(v)
        return filtered_vars
    
    def filter_ns_vars(
            self, var_set: set, allow_bool: bool = True, allow_cont: bool = True
    ) -> Set[str]:
        filtered_vars = set()
        for v in var_set:
            if allow_cont and (v in self.mdp._cont_ns_vars):
                filtered_vars.add(v)
            elif allow_bool and (v in self.mdp._bool_ns_vars):
                filtered_vars.add(v)
        return filtered_vars
    
    def filter_i_vars(
            self, var_set: set, allow_bool: bool = True, allow_cont: bool = True
    ) -> Set[str]:
        filtered_vars = set()
        for v in var_set:
            if allow_cont and (v in self.mdp._cont_i_vars):
                filtered_vars.add(v)
            elif allow_bool and (v in self.mdp._bool_i_vars):
                filtered_vars.add(v)
        return filtered_vars

    def rank_vars(self, var_set):
        level_dict_reverse = {}
        return

            


        
        
    def flush_caches(self):
        pass
    
    def make_XADD_label(self):
        pass
    
    def save_results(self):
        pass

    def print(self, node_id: int):
        print(self.context.get_repr(node_id))

    @property
    def context(self):
        return self.mdp.context
