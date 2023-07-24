from modeldiff.diffenv.diffClass import ModelDiffReservoir

import json

## experiment 1
policy_path = 'policies/res/'
m1_path = "/home/jackliu/model-diff/model_diff_DQN/RDDL/Reservoir_disc_1res_source/domain.rddl"
m2_path = "/home/jackliu/model-diff/model_diff_DQN/RDDL/Reservoir_disc_1res_target/domain.rddl"

ModelDiff = ModelDiffReservoir(m1_path, m2_path, policy_path)
ModelDiff.build_model_with_diff_reward()

model_1 = ModelDiff._model_1
context_1 = ModelDiff._context_1

model_2 = ModelDiff._model_2
context_2 = ModelDiff._context_2

model_diff = ModelDiff._model_diff
context_diff = ModelDiff._context_diff

reward_node_1 = context_1._id_to_node.get(model_1.reward)
print("Reward XADD 1")
print(reward_node_1)

reward_node_2 = context_2._id_to_node.get(model_2.reward)
print("Reward XADD 2")
print(reward_node_2)

reward_node_diff = context_diff._id_to_node.get(model_diff.reward)
print("Reward XADD Diff")
print(reward_node_diff)

print('MDP1')
vid_1, q_1 = ModelDiff.do_PE(model_1, context_1, 0.9, 2)

print('MDP2')
vid_2, q_2 = ModelDiff.do_PE(model_2, context_2, 0.9, 2)

print('MDP_diff')
vid_diff, q_diff = ModelDiff.do_PE(model_diff, context_diff, 0.9, 2)

def save_value_function(fpath, name, node_id, context):
    node = context._id_to_node.get(node_id, None)
    node.turn_off_print_node_info()
    node_str = str(node)
    node_dict = {name:node_str}
    with open(fpath+name+'.json', 'w') as f:
        json.dump(node_dict, f)

def save_q_function(fpath, name, q_list, context):
    q_xadd_list = []
    for i in q_list:
        action = i[0]
        node_id = i[1]
        node = context._id_to_node.get(node_id, None)
        node.turn_off_print_node_info()
        node_str = str(node)
        q_xadd_list.append((action, node_str))
    with open(fpath+name+'.json', 'w') as f:
        json.dump(q_xadd_list, f)

# save_value_function('/home/jackliu/model-diff/model_diff_DQN/value_xadd/1res/', 'v_source', vid_1, context_1)
# save_value_function('/home/jackliu/model-diff/model_diff_DQN/value_xadd/1res/', 'v_target', vid_2, context_2)
# save_value_function('/home/jackliu/model-diff/model_diff_DQN/value_xadd/1res/', 'v_diff', vid_diff, context_diff)

# save_q_function('/home/jackliu/model-diff/model_diff_DQN/value_xadd/1res/', 'q_source', q_1, context_1)
# save_q_function('/home/jackliu/model-diff/model_diff_DQN/value_xadd/1res/', 'q_target', q_2, context_2)
# save_q_function('/home/jackliu/model-diff/model_diff_DQN/value_xadd/1res/', 'q_diff', q_diff, context_diff)


# context_2.export_xadd(vid_2, './exported_xadds/1res_v2.xadd')
# for i in q_2:
#     k = list(i[0].keys())[0]
#     v = i[0][k]
#     xadd_id = i[1]
#     context_2.export_xadd(xadd_id, './exported_xadds/1res_v2_{}_{}.xadd'.format(k,v))

# context_diff.export_xadd(vid_diff, './exported_xadds/1res_vdiff.xadd')
# for i in q_diff:
#     k = list(i[0].keys())[0]
#     v = i[0][k]
#     xadd_id = i[1]
#     context_diff.export_xadd(xadd_id, './exported_xadds/1res_vdiff_{}_{}.xadd'.format(k,v))


# print(context_1._id_to_node.get(vid_1))
# print(context_2._id_to_node.get(vid_2))
# print(context_diff._id_to_node.get(vid_diff))

# test_dict_c_1 = {'rlevel___t1':0}
# test_dict_c_2 = {'rlevel___t1':0}
# test_dict_c_diff = {'rlevel___t1':0}
# test_dict_b = {}

# result_list_1 = []
# result_list_2 = []
# result_list_diff = []
# for i in range(0, 101, 1):
#     test_dict_c_1['rlevel___t1'] = i
#     test_dict_c_2['rlevel___t1'] = i
#     test_dict_c_diff['rlevel___t1'] = i
#     v_1 = ModelDiff.eval_function(test_dict_b, test_dict_c_1, vid_1, model_1, context_1)
#     v_2 = ModelDiff.eval_function(test_dict_b, test_dict_c_2, vid_2, model_2, context_2)
#     v_diff = ModelDiff.eval_function(test_dict_b, test_dict_c_diff, vid_diff, model_diff, context_diff)
#     print(i, v_1, v_2, v_diff, v_1 + v_diff == v_2)

# test_dict_c_1 = {'rlevel___t1':0,'rlevel___t2':0}
# test_dict_c_2 = {'rlevel___t1':0,'rlevel___t2':0}
# test_dict_c_diff = {'rlevel___t1':0,'rlevel___t2':0}
# test_dict_b = {}

# result_list_1 = []
# result_list_2 = []
# result_list_diff = []
# for i in range(0, 101, 5):
#     for j in range(0, 101, 5):
#         test_dict_c_1['rlevel___t1'] = i
#         test_dict_c_1['rlevel___t2'] = j
#         test_dict_c_2['rlevel___t1'] = i
#         test_dict_c_2['rlevel___t2'] = j
#         test_dict_c_diff['rlevel___t1'] = i
#         test_dict_c_diff['rlevel___t2'] = j
#         v_1 = ModelDiff.eval_function(test_dict_b, test_dict_c_1, vid_1, model_1, context_1)
#         v_2 = ModelDiff.eval_function(test_dict_b, test_dict_c_2, vid_2, model_2, context_2)
#         v_diff = ModelDiff.eval_function(test_dict_b, test_dict_c_diff, vid_diff, model_diff, context_diff)
#         print(i, v_1, v_2, v_diff, v_1 + v_diff == v_2)
