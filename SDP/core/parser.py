from pyRDDLGym.XADD.RDDLModelXADD import RDDLModelWXADD

from SDP.core.action import Action
from SDP.core.mdp import MDP

import itertools


class Parser:
    def __init__(self):
        pass    

    def parse(
            self,
            model: RDDLModelWXADD,
            is_linear: bool = False
    ) -> MDP:
        """Parses the RDDL model into an MDP."""
        mdp = MDP(model, is_linear=is_linear)

        # Go through all actions and get corresponding CPFs and rewards
        actions = model.actions
        action_dict = {}
        action_type = {}
        for name, val in actions.items():
            atype = 'bool' if isinstance(val, bool) else 'real'
            a_symbol = model.ns.get(name)
            if a_symbol is None:
                print(f'Warning: action {name} not found in RDDLModelWXADD.actions')
                a_symbol, a_node_id = model.add_sympy_var(name, atype)
            action_dict[a_symbol] = False
            action_type[name] = atype
        
        for name, val in actions.items():
            atype = action_type[name]
            a_symbol = model.ns[name]
            action = Action(
                name, a_symbol, mdp.context, atype=atype, action_params=None
            )    # TODO: action_params (for continuous actions)

            # Get the cpfs corresponding to the action
            subst_dict = action_dict.copy()
            subst_dict[a_symbol] = True
            cpfs = model.cpfs
            for state_fluent, cpf in cpfs.items():
                cpf = action.restrict(cpf, subst_dict)
                var_ = model.ns[state_fluent]
                action.add_cpf(var_, cpf)
            
            # Get the reward corresponding to the action
            reward = model.reward
            reward = action.restrict(reward, subst_dict)
            action.reward = reward

            mdp.add_action(action)
        
        return mdp