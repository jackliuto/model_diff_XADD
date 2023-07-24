
from model_generation.diffenv.diffClass import ModelDiff
from utils.xadd_utils import *

import json

params = Params("xadd_params.json")

ModelDiff = ModelDiff(params.domain_type, params.model_source_path, params.model_target_path, params.policy_path)
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
print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')

vid_1, q_1 = ModelDiff.do_PE(model_1, context_1, 1, 2)

raise ValueError

vid_2, q_2 = ModelDiff.do_PE(model_2, context_2, 0.9, 2)
vid_diff, q_diff = ModelDiff.do_PE(model_diff, context_diff, 0.9, 2)

save_value_function(params.save_path, 'v_source', vid_1, context_1)
save_value_function(params.save_path, 'v_target', vid_2, context_2)
save_value_function(params.save_path, 'v_diff', vid_diff, context_diff)

save_q_function(params.save_path, 'q_source', q_1, context_1)
save_q_function(params.save_path, 'q_target', q_2, context_2)
save_q_function(params.save_path, 'q_diff', q_diff, context_diff)

# for i in q_1:
#     print(i[0])
#     print(context_1._id_to_node.get(i[1]))

# print(context_1._id_to_node.get(vid_1))

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
