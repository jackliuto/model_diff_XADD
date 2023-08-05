from typing import Dict

from SDP.core.mdp import MDP
from SDP.core.action import Action


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

    @property
    def context(self):
        return self.mdp.context
