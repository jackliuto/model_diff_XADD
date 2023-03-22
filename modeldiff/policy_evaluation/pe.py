from typing import Set
import sympy as sp

import xaddpy
from pyRDDLGym.XADD.RDDLModelXADD import RDDLModelWXADD
from xaddpy.xadd.xadd import XADD, DeltaFunctionSubstitution

from modeldiff.core.base import Action
from modeldiff.core.mdp import MDP
from modeldiff.utils.global_vars import SUM, PROD, RESTRICT_HIGH, RESTRICT_LOW


class PolicyEvaluation:

    def __init__(self, mdp: MDP, iter: int):
        self.mdp = mdp
        self._cur_iter = 0
        self._n_iter = iter
        self._res_dd = 0
        self._prev_dd = 0
    
    def solve(self):
        # Initialize the iteration counter
        self._cur_iter = 0

        # Initialize the value function to be the zero node
        value_dd = self.context.ZERO

        # Perform policy evaluation for the specified number of iterations, or until convergence
        while self._cur_iter < self._n_iter:
            # Compute the next value function
            value_dd = self.bellman_backup(value_dd)

            # Increment the iteration counter
            self._cur_iter += 1

            self._prev_dd = value_dd
            self.bellman_backup(value_dd)
    
    def bellman_backup(self, value_dd: int) -> int:
        """Performs a single iteration of the Bellman backup.
        
        Args:
            value_dd (int): The current value function.
        
        Returns:
            int: The next value function.
        """
        res_dd = value_dd

        # Iterate over all actions
        for aname, action in self.mdp.actions.items():
            regr = self.regress(res_dd, action)
        
        return res_dd
    
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

    def regress_c_vars(self, q: int, a: Action, v: str) -> int:
        # Get the cpf for the variable
        cpf = a.get_cpf(v)
        
        # Check regression cache
        key = (v, cpf, q)
        res = self.mdp._cont_regr_cache.get(key)
        if res is not None:
            return res

        # Perform regression via delta function substitution
        v_ = self.mdp.model.ns[v]
        leaf_op = DeltaFunctionSubstitution(v_, q, self.context)
        q = self.context.reduce_process_xadd_leaf(cpf, leaf_op, [], [])
        
        # Cache result
        self.mdp._cont_regr_cache[key] = q

        return q
    
    def regress_b_vars(self, q: int, a: Action, v: str) -> int:
        # Get the cpf for the variable
        cpf = a.get_cpf(v)
        dec_id = self.context._expr_to_id[self.mdp.model.ns[v]]

        # Marginalize out the variable
        q = self.context.apply(q, cpf, PROD)

        restrict_high = self.context.op_out(q, dec_id, RESTRICT_HIGH)
        restrict_low = self.context.op_out(q, dec_id, RESTRICT_LOW)
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

    @property
    def context(self):
        return self.mdp.context
