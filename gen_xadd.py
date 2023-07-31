
from model_generation.diffenv.diffClass import ModelDiff
from utils.xadd_utils import *

import pathlib

import json

params = Params("./params/xadd_params_inventory.json")

ModelDiff = ModelDiff(domain_type=params.domain_type, 
                      domain_path=params.domain_path, 
                        instance_source_path=params.instance_source_path, 
                        instance_target_path=params.instance_target_path, 
                        policy_path=params.policy_path)
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

vid_1, q_1 = ModelDiff.do_PE(model_1, context_1, params.discount_rate, params.horizon_length)
vid_2, q_2 = ModelDiff.do_PE(model_2, context_2, params.discount_rate, params.horizon_length)
vid_diff, q_diff = ModelDiff.do_PE(model_diff, context_diff, params.discount_rate, params.horizon_length)

print("Context 1 Nodes: ", len(context_1._id_to_node))
print("Context 2 Nodes: ",len(context_2._id_to_node))
print("Context diff Nodes: ",len(context_diff._id_to_node))

print(context_1._id_to_node[vid_1])

## testing for values
# test_dict_c_1 = {'rlevel___t1':0}
test_dict_c_1 = {'stock___i1':37}
test_dict_b = {}
result_list_1 = []
v_1_v = ModelDiff.eval_function(test_dict_b, test_dict_c_1, vid_1, model_1, context_1)
print(v_1_v)


### testing for values
# # test_dict_c_1 = {'rlevel___t1':0}
# test_dict_c_1 = {'stock___i1':0}
# test_dict_b = {}
# result_list_1 = []
# v_1_v = ModelDiff.eval_function(test_dict_b, test_dict_c_1, vid_1, model_1, context_1)




if params.save_xadds:
  xadd_path = pathlib.Path(params.save_path+'{}_step/'.format(params.horizon_length))
  xadd_path.mkdir(parents=True, exist_ok=True)

  save_value_function(params.save_path+'{}_step/'.format(params.horizon_length), 'v_source', vid_1, context_1)
  save_value_function(params.save_path+'{}_step/'.format(params.horizon_length), 'v_target', vid_2, context_2)
  save_value_function(params.save_path+'{}_step/'.format(params.horizon_length), 'v_diff', vid_diff, context_diff)

  save_q_function(params.save_path+'{}_step/'.format(params.horizon_length), 'q_source', q_1, context_1)
  save_q_function(params.save_path+'{}_step/'.format(params.horizon_length), 'q_target', q_2, context_2)
  save_q_function(params.save_path+'{}_step/'.format(params.horizon_length), 'q_diff', q_diff, context_diff)

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

###############################################

# test_dict_c_1 = {'stock___i1':0}
# test_dict_c_2 = {'stock___i1':0}
# test_dict_c_diff = {'stock___i1':0}
# test_dict_b = {}

# result_list_1 = []
# result_list_2 = []
# result_list_diff = []

# for i in range(0, 51, 1):
#     test_dict_c_1['stock___i1'] = i
#     test_dict_c_2['stock___i1'] = i
#     test_dict_c_diff['stock___i1'] = i
#     q_1_v = ModelDiff.eval_function(test_dict_b, test_dict_c_1, q_1[0][1], model_1, context_1)
#     q_2_v = ModelDiff.eval_function(test_dict_b, test_dict_c_2, q_2[0][1], model_2, context_2)
#     q_diff_v = ModelDiff.eval_function(test_dict_b, test_dict_c_diff, q_diff[0][1], model_diff, context_diff)
#     v_1_v = ModelDiff.eval_function(test_dict_b, test_dict_c_1, vid_1, model_1, context_1)
#     v_2_v = ModelDiff.eval_function(test_dict_b, test_dict_c_2, vid_2, model_1, context_2)
#     print(i, v_1_v, v_2_v)
#     # print(i, q_1_v, q_diff_v, q_2_v, q_1_v + q_diff_v - q_2_v, q_1_v + q_diff_v == q_2_v)


# ## Tests q values to see if they are the same

# test_dict_c_1 = {'stock___i1':0,'stock___i2':0}
# test_dict_c_2 = {'stock___i1':0,'stock___i2':0}
# test_dict_c_diff = {'stock___i1':0,'stock___i2':0}
# test_dict_b = {}

# result_list_1 = []
# result_list_2 = []
# result_list_diff = []

# max_error = 0

# for n in range(4):
#     for i in range(0, 51, 2):
#         for j in range(0, 51, 2):
#             test_dict_c_1['stock___i1'] = i
#             test_dict_c_1['stock___i2'] = j
#             test_dict_c_2['stock___i1'] = i
#             test_dict_c_2['stock___i2'] = j
#             test_dict_c_diff['stock___i1'] = i
#             test_dict_c_diff['stock___i2'] = j
#             q_1_v = ModelDiff.eval_function(test_dict_b, test_dict_c_1, q_1[n][1], model_1, context_1)
#             q_2_v = ModelDiff.eval_function(test_dict_b, test_dict_c_2, q_2[n][1], model_2, context_2)
#             q_diff_v = ModelDiff.eval_function(test_dict_b, test_dict_c_diff, q_diff[n][1], model_diff, context_diff)
#             print(i,j, q_1_v, q_2_v, q_diff_v, q_1_v + q_diff_v,  q_1_v + q_diff_v - q_2_v, q_1_v + q_diff_v == q_2_v)
#             if max_error < abs(q_1_v + q_diff_v - q_2_v):
#                 max_error = abs(q_1_v + q_diff_v - q_2_v)

# print(max_error)
