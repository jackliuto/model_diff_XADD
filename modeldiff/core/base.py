from pyRDDLGym.XADD.RDDLModelXADD import RDDLModelWXADD


class Action:
    def __init__(
            self, 
            action_name: str, 
            arity: int, 
            model: RDDLModelWXADD,
            atype: str,
            action_params: dict = None
    ):
        self._name = action_name
        self._arity = arity
        self._model = model
        self._atype = atype         # action type: 'boolean' or 'continuous'
        self._action_params = action_params
        self._cpfs = {}
    
    def restrict(self, cpf: int):
        """Restricts the CPF to this particular action"""
        context = self._model._context
        if self._atype == 'boolean':
            subst_dict = {self._name: True}
            return context.substitute(cpf, subst_dict)
        else:
            raise NotImplementedError("Continuous actions are not supported yet")

    def add_cpf(self, name: str, cpf: int):
        """Adds a CPF to this action"""
        self._cpfs[name] = cpf
    
    def get_cpf(self, name: str) -> int:
        """Gets the CPF corresponding to the given name"""
        return self._cpfs[name]

    @property
    def reward(self) -> int:
        return self._reward
    
    @reward.setter
    def reward(self, reward: int):
        self._reward = reward

    def __repr__(self) -> str:
        return self._name
    