from typing import Dict

from SDP.core.mdp import MDP
from SDP.core.action import Action

import sympy as sp


class Policy:
    """A policy class to support policy evaluation"""
    def __init__(self, mdp: MDP):
        self.mdp = mdp
        self._dist: Dict[Action, int] = {}
    
    def get_policy_xadd(self, action: Action) -> int:
        """Returns the policy XADD for the specified action.
        
        Args:
            action (Action): The action.
        
        Returns:
            int: The policy DD.
        """
        return self._dist.get(action)

    def load_policy(self, policy: Dict[Action, int]):
        """Loads an XADD per each action.
        
        Args:
            policy (Dict[Action, int]): The policy.
        """
        for action, node_id in policy.items():
            self._dist[action] = node_id
    
    def compile_policy(self):
        policy_id_true = self.mdp.context.ONE
        policy_id_false = self.mdp.context.ONE

        for action, policy_id in self._dist.items():
            if action._atype != 'bool':
                raise NotImplementedError("Continuous actions are not supported yet")
            
            # get action in int form
            dec_id, is_reversed = self.mdp.context.get_dec_expr_index(action._symbol, create=True)
            high: int = self.mdp.context.get_leaf_node(sp.S(1))
            low: int = self.mdp.context.get_leaf_node(sp.S(0))
            if is_reversed:
                low, high = high, low
            action_id: int = self.mdp.context.get_internal_node(dec_id, low=low, high=high)
            
            # generate true and false policy
            action_policy_id_true = self.mdp.context.apply(policy_id, action_id, 'prod')
            action_policy_id_false = self.mdp.context.apply(self.mdp.context.ONE, action_policy_id_true, 'false')

            policy_id_true = self.mdp.context.apply(action_policy_id_true, action_policy_id_true, 'max')
            policy_id_false = self.mdp.context.apply(action_policy_id_false, action_policy_id_false, 'max')

        



            
            

        # policy_id = self.mdp.context.ONE
        # action_cpf = self.mdp._model.cpfs
        # print(self.mdp._model._var_name_to_node_id['move'])
        # print(self.mdp.context.get_repr(self.mdp._model._var_name_to_node_id['move']))
        # a = self.mdp.context.apply(policy_id, self.mdp._model._var_name_to_node_id['move'], 'add')
        # print(self.mdp.context.get_repr(a))
        # for action, a_id in self._dist.items():
        #     pass
        #     print(action, a_id)
        #     print(self.mdp.context.get_repr(a_id))
        #     print(self.mdp.model.actions)
        # return




    @property
    def context(self):
        return self.mdp.context
