import sympy as sp
from typing import Dict, Tuple
from pyRDDLGym.XADD.RDDLModelXADD import RDDLModelWXADD
from pyRDDLGym.XADD.RDDLLevelAnalysisXADD import RDDLLevelAnalysisWXADD

from SDP.core.action import Action


class MDP:
    def __init__(
            self, 
            model: RDDLModelWXADD, 
            is_linear: bool = False, 
            discount: float = 1.0
    ):
        self._model = model

        self._levelanalysis = RDDLLevelAnalysisWXADD(model)

        print(self._levelanalysis.build_call_graph())

        self._is_linear = is_linear
        self._discount = discount
        self._prime_subs = self.get_prime_subs()
        self._cont_ns_vars = set()
        self._bool_ns_vars = set()
        self._cont_i_vars = set()
        self._bool_i_vars = set()
        self._bool_s_vars = set()   # TODO: necessary?
        self._cont_s_vars = set()   # TODO: necessary?
        self._cont_a_vars = set()   # This might be redundant
        
        # Cache
        self._cont_regr_cache: Dict[Tuple[str, int, int], int] = {}

        self._actions: Dict[str, Action] = {}
        self.update_var_sets()

    def get_prime_subs(self) -> Dict[sp.Symbol, sp.Symbol]:
        model = self._model
        s_to_ns = model.next_state
        prime_subs = {}
        for s, ns in s_to_ns.items():
            s_var = model.ns[s]
            ns_var, var_node_id = model.add_sympy_var(ns, model.gvar_to_type[ns])
            prime_subs[s_var] = ns_var
        return prime_subs

    def update_var_sets(self):
        model = self._model
        for v, vtype in model.gvar_to_type.items():
            var_, var_node_id = model.add_sympy_var(v, vtype)
            if v in model.next_state.values():
                if vtype == 'bool':
                    self._bool_ns_vars.add(var_)
                else:
                    self._cont_ns_vars.add(var_)
            elif v in model.interm:
                if vtype == 'bool':
                    self._bool_i_vars.add(var_)
                else:
                    self._cont_i_vars.add(var_)
        
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
    def discount(self):
        return self._discount
    
    @property
    def model(self):
        return self._model

    @property
    def cpfs(self):
        return self._model.cpfs

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
