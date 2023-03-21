from pyRDDLGym.XADD.RDDLModelXADD import RDDLModelWXADD
from modeldiff.core.base import Action
from modeldiff.core.mdp import MDP

class Parser:
    def __init__(self):
        pass    

    def parse(self, model: RDDLModelWXADD) -> MDP:
        mdp = MDP(model)

        # Go through all actions and get corresponding CPFs and rewards
        actions = model.actions
        
        for name, arity in actions.items():
            action = Action(name, arity, action_params=None)    # TODO: action_params (for continuous actions)

            # Get the cpfs corresponding to the action
            cpfs = self._model.cpfs
            for state_fluent, cpf in cpfs.items():
                cpf = action.restrict(cpf)
                action.add_cpf(state_fluent, cpf)
            
            # Get the reward corresponding to the action
            reward = self._model.reward
            reward = action.restrict(reward)
            action.reward = reward

            mdp.add_action(action)
        
        return mdp
