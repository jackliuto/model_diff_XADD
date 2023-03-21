from typing import Dict
from pyRDDLGym.XADD.RDDLModelXADD import RDDLModelWXADD

from modeldiff.core.base import Action


class MDP:
    def __init__(self, model: RDDLModelWXADD, is_linear: bool = False):
        self._model = model
        self._is_linear = is_linear
        self._prime_subs = self._model.next_state
        self._cont_ns_vars = set()
        self._bool_ns_vars = set()
        self._cont_i_vars = set()
        self._bool_i_vars = set()
        self._bool_s_vars = set()
        self._cont_s_vars = set()
        self._cont_a_vars = set()   # This might be redundant
        
        self._actions: Dict[str, Action] = {}
        self.update_var_sets()

    def update_var_sets(self):
        model = self._model
        for v, vtype in model.gvar_to_type.items():
            if v in model.next_state.values():
                if vtype == 'bool':
                    self._bool_ns_vars.add(v)
                else:
                    self._cont_ns_vars.add(v)
            elif v in model.interm_state():
                if vtype == 'bool':
                    self._bool_i_vars.add(v)
                else:
                    self._cont_i_vars.add(v)
        
    def add_action(self, action: Action):
        self._actions[action._name] = action
    
    def standardize_dd(self, node_id: int) -> int:
        node_id = self.context.make_canonical(node_id)
        if self._is_linear:
            node_id = self.context.reduce_lp(node_id)
        self.check_standard_dd(node_id)
        return node_id

    def check_standard_dd(self, node_id: int) -> bool:
        standard = True
        if not self.check_canon(node_id):
            standard = False
        if self._is_linear and not self.check_reduce_lp(node_id):
            standard = False
        return standard
    
    def check_canon(self, node_id: int) -> bool:
        canon_dd = self.context.make_canonical(node_id)
        if node_id != canon_dd:
            node, canon_node = self.context.get_exist_node(node_id), self.context.get_exist_node(canon_dd)
            node_size, canon_size = len(node.collect_nodes()), len(canon_node.collect_nodes())
            print(f"Check canon failed for node {node_id} with size {node_size}, Canon DD size: {canon_size}")
            return False
        return True        
    
    def check_reduce_lp(self, node_id: int) -> bool:
        reduced_node_id = self.context.reduce_lp(node_id)
        if (node_id != reduced_node_id):
            node, reduced_node = self.context.get_exist_node(node_id), self.context.get_exist_node(reduced_node_id)
            node_size, reduced_size = len(node.collect_nodes()), len(reduced_node.collect_nodes())
            print(f"Check reduce lp failed for node {node_id} with size {node_size}, Reduced DD size: {reduced_size}")
            return False
        return True
    
    @property
    def context(self):
        return self._model._context

    @property
    def actions(self):
        return self._actions

    @property
    def reward(self) -> int:
        return self._model.reward
    
    @property
    def prime_subs(self):
        return self._prime_subs
