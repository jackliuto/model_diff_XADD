from typing import Dict, Union
import sympy as sp
from xaddpy.xadd.xadd import XADD


class Action:
    def __init__(
            self, 
            name: str,
            symbol: sp.Symbol,
            bool_dict: dict,
            context: XADD,
            atype: str,
            action_params: dict = None
    ):
        self._name = name
        self._symbol = symbol
        self._bool_dict = bool_dict
        self._context = context
        self._atype = atype         # action type: 'bool' or 'real'
        self._action_params = action_params
        self._cpfs = {}
    
    def restrict(self, cpf: int, subst_dict: Dict[sp.Symbol, Union[bool, float, int]] = {}, bool_val=True):
        """Restricts the CPF to this particular action"""
        context = self._context
        if self._atype == 'bool':
            # subst_dict.update({self._symbol: bool_val})
            # subst_dict.update({self._symbol: 1.0})
            pass
            return context.substitute(cpf, subst_dict)
        else:
            raise NotImplementedError("Continuous actions are not supported yet")

    def add_cpf(self, v: sp.Symbol, cpf: int):
        """Adds a CPF to this action"""
        self._cpfs[v] = cpf
    
    def get_cpf(self, v: sp.Symbol) -> int:
        """Gets the CPF corresponding to the given name"""
        return self._cpfs[v]

    @property
    def reward(self) -> int:
        return self._reward
    
    @reward.setter
    def reward(self, reward: int):
        self._reward = reward

    # def __repr__(self) -> str:
    #     return self._name
    