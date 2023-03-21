from xaddpy.xadd.xadd import XADD
import sympy as sp

context = XADD()
x, y = sp.S('x'), sp.S('y')
east, west, north, south = sp.S('east'), sp.S('west'), sp.S('north'), sp.S('south'),

def buid_xadd_list(x,y):
    GOAL_X = x
    GOAL_Y = y

    expr_sm_x = GOAL_X - x > 0
    expr_sm_y = GOAL_Y - y > 0

    expr_gt_x = GOAL_X - x < 0
    expr_gt_y = GOAL_Y - y < 0

    xadd_as_list = [expr_sm_x, 
                        [east], 
                        [expr_gt_x, 
                            [west], 
                            [expr_sm_y, 
                                [north], 
                                [expr_gt_y, 
                                    [south], 
                                    [sp.S(0)]
                                ]
                            ]
                        ]
                    ]

    return xadd_as_list

def buid_xadd_list_two_actions(v):
    GOAL_X = v
    GOAL_Y = v

    expr_sm_x = GOAL_X - x > 0
    expr_sm_y = GOAL_Y - y > 0

    expr_gt_x = GOAL_X - x < 0
    expr_gt_y = GOAL_Y - y < 0

    xadd_as_list = [expr_sm_x, 
                        [east], 
                        [expr_sm_y, 
                            [west], 
                            [sp.S(0)]
                            ]
                    ]

    return xadd_as_list

def buid_xadd_list_1d(v):
    GOAL_X = v


    expr_sm_x = GOAL_X - x > 0
    expr_gt_x = GOAL_X - x < 0

    xadd_as_list = [expr_sm_x, 
                        [sp.S(1)], 
                        [sp.S(0)]
                    ]

    return xadd_as_list

# xadd_list1 = buid_xadd_list_two_actions(10)
# xadd_list2 = buid_xadd_list_two_actions(5)

xadd_list1 = buid_xadd_list_two_actions(10)
xadd_list2 = buid_xadd_list_two_actions(5)


n1 = context.build_initial_xadd(xadd_list1)
n2 = context.build_initial_xadd(xadd_list2)

n3 = context.apply(n1, n2, 'subtract')
context.reduce_lp(n3)


context.save_graph(n3, './n3_2')
context.save_graph(n1, './n1_2')
context.save_graph(n2, './n2_2')

print(context.get_exist_node(n3))

