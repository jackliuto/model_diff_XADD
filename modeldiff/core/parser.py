import sympy as sp
from pyRDDLGym.XADD.RDDLModelXADD import RDDLModelWXADD
from modeldiff.core.action import Action
from modeldiff.core.mdp import MDP

class Parser:
    def __init__(self):
        pass    

    def parse(self, model: RDDLModelWXADD, is_linear: bool = False) -> MDP:
        mdp = MDP(model, is_linear=is_linear)

        # Go through all actions and get corresponding CPFs and rewards
        actions = model.actions
        
        for name, val in actions.items():
            atype = 'bool' if isinstance(val, bool) else 'real'
            a_symbol = model.ns.get(name)
            if a_symbol is None:
                print(f'Warning: action {name} not found in RDDLModelWXADD.actions')
                a_symbol = model.add_sympy_var(name, atype)
            action = Action(
                name, a_symbol, mdp.context, atype=atype, action_params=None
            )    # TODO: action_params (for continuous actions)

            # Get the cpfs corresponding to the action
            cpfs = model.cpfs
            for state_fluent, cpf in cpfs.items():
                cpf = action.restrict(cpf)
                var_ = model.ns[state_fluent]
                action.add_cpf(var_, cpf)
            
            # Get the reward corresponding to the action
            reward = model.reward
            reward = action.restrict(reward)
            action.reward = reward

            mdp.add_action(action)
        
        return mdp
