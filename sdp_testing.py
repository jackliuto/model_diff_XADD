import sympy as sp
from xaddpy import XADD

context = XADD()

# Get the unique ID of the decision expression
dec_expr: sp.Basic = sp.S('x + y <= 0')
dec_id, is_reversed = context.get_dec_expr_index(dec_expr, create=True)

# Get the IDs of the high and low branches: [0] and [2], respectively
high: int = context.get_leaf_node(sp.S(1))
low: int = context.get_leaf_node(sp.S(2))
if is_reversed:
    low, high = high, low

# Create the decision node with the IDs
dec_node_id: int = context.get_internal_node(dec_id, low=low, high=high)
print(f"Node created:\n{context.get_repr(dec_node_id)}")

b = sp.Symbol('b', bool=True)
dec_b_id, _ = context.get_dec_expr_index(b, create=True)
high: int = context.get_leaf_node(sp.S(True))
low: int = context.get_leaf_node(sp.S(False))
dec_bnode_id: int = context.get_internal_node(dec_b_id, low=low, high=high)
print(context.get_repr(dec_bnode_id))

sum_node_id = context.apply(dec_node_id, dec_bnode_id, op='min')


print(context.get_repr(sum_node_id))

