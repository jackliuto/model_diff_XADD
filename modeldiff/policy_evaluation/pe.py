from typing import Set
import sympy as sp

import xaddpy
from pyRDDLGym.XADD.RDDLModelXADD import RDDLModelWXADD
from xaddpy.xadd.xadd import XADD, DeltaFunctionSubstitution

from modeldiff.core.action import Action
from modeldiff.core.mdp import MDP
from modeldiff.core.policy import Policy
from modeldiff.utils.global_vars import SUM, PROD, RESTRICT_HIGH, RESTRICT_LOW


class PolicyEvaluation:

    def __init__(self, mdp: MDP, policy: Policy, iter: int):
        """Performs policy evaluation for the specified number of iterations, or until convergence.
        
        Args:
            mdp (MDP): The MDP.
            policy (Policy): The policy to evaluate.
            iter (int): The number of iterations to perform.
        """
        self.mdp = mdp
        self.policy = policy
        self._cur_iter = 0
        self._final_iter = 0
        self._n_iter = iter
        self._res_dd = 0
        self._prev_dd = 0
    
    def solve(self) -> int:
        # Initialize the iteration counter
        self._cur_iter = 0

        # Initialize the value function to be the zero node
        value_dd = self.context.ZERO

        # Perform policy evaluation for the specified number of iterations, or until convergence
        while self._cur_iter < self._n_iter:

            self._prev_dd = value_dd
            
            # Compute the next value function
            value_dd, q_list = self.bellman_backup(value_dd)

            # Increment the iteration counter
            self._cur_iter += 1
        self._final_iter = self._cur_iter
        return value_dd, q_list
    
    def bellman_backup(self, value_dd: int) -> int:
        """Performs a single iteration of the Bellman backup.
        
        Args:
            value_dd (int): The current value function.
        
        Returns:
            int: The next value function.
        """
        res_dd = self.context.ZERO      # Accumulate the value function in this variable

        q_list = []
        

        # Iterate over all actions
        for aname, action in self.mdp.actions.items():
            # Compute the action value function
            regr = self.regress(value_dd, action)

            # get q value 
            q_list.append((action._bool_dict, regr))

            # Multiply by pi(a|s)
            # Note: since everything's symbolic, state is not specified
            regr = self.context.apply(regr, self.policy.get_policy_xadd(action), PROD)
 
            if self.mdp._is_linear:
                regr = self.context.reduce_lp(regr)
            
            res_dd = self.context.apply(regr, res_dd, SUM)
            
        return res_dd, q_list
    
    def regress(self, value_dd: int, action: Action, regress_cont: bool = False) -> int:
        # Prime the value function
        subst_dict = self.mdp.prime_subs

        q = self.context.substitute(value_dd, subst_dict)

        # Discount
        if self.mdp.discount < 1.0:
            q = self.context.scalar_op(q, self.mdp.discount, PROD)

        # Add reward *if* it contains primed vars that need to be regressed
        i_and_ns_vars_in_reward = self.filter_i_and_ns_vars(self.context.collect_vars(action.reward))

        if len(i_and_ns_vars_in_reward) > 0:
            q = self.context.apply(q, action.reward, SUM)
        
        # Get variables to eliminate
        # TODO: Do we need to handle topological ordering?
        # graph = self.mdp.build_dbn_dependency_dag(action, vars_to_regress)        
        vars_to_regress = self.filter_i_and_ns_vars(self.context.collect_vars(q), True, True)
        
        # Regress each variable
        for v in vars_to_regress:
            if v in self.mdp._cont_ns_vars or v in self.mdp._cont_i_vars:
                q = self.regress_c_vars(q, action, v)
            elif v in self.mdp._bool_ns_vars or v in self.mdp._bool_i_vars:
                q = self.regress_b_vars(q, action, v)
        
        # Add the reward
        if len(i_and_ns_vars_in_reward) == 0:
            q = self.context.apply(q, action.reward, SUM)

        # Continuous noise?
        # TODO
        # q = self.regress_noise(q, action)

        # Continuous parameter
        if regress_cont:
            q = self.regress_action(q, action)

        q = self.mdp.standardize_dd(q)      

        return q

    def regress_c_vars(self, q: int, a: Action, v: sp.Symbol) -> int:
        # Get the cpf for the variable
        cpf = a.get_cpf(v)
        
        # Check regression cache
        key = (str(v), cpf, q)
        res = self.mdp._cont_regr_cache.get(key)
        if res is not None:
            return res

        # Perform regression via delta function substitution
        leaf_op = DeltaFunctionSubstitution(v, q, self.context)

        q = self.context.reduce_process_xadd_leaf(cpf, leaf_op, [], [])
        if self.mdp._is_linear:
            q = self.context.reduce_lp(q)
        
        # Cache result
        self.mdp._cont_regr_cache[key] = q

        return q
    
    def regress_b_vars(self, q: int, a: Action, v: str) -> int:
        # Get the cpf for the variable
        cpf = a.get_cpf(v)

        dec_id = self.context._expr_to_id[self.mdp.model.ns[str(v)]]


        # # Marginalize out the variable uncomment for the original,
        # q = self.context.apply(q, cpf, PROD)
        
        restrict_high = self.context.op_out(q, dec_id, RESTRICT_HIGH)
        restrict_low = self.context.op_out(q, dec_id, RESTRICT_LOW)

        # # # Handcrafted marginalization
        # prob = float(str(self.context._id_to_node[cpf]).split()[3])
        # true_prop_id = self.context.get_leaf_node(sp.S(prob))
        # false_prop_id = self.context.get_leaf_node(sp.S(1 - prob))
        # restrict_high = self.context.apply(restrict_high, true_prop_id, PROD)
        # restrict_low = self.context.apply(restrict_low, false_prop_id, PROD)  

        q = self.context.apply(restrict_high, restrict_low, SUM)

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
