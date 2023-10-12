
from model_generation.diffenv.diffClass import ModelDiff
from utils.xadd_utils import *

import pathlib

import json

import matplotlib.pyplot as plt
import numpy as np

import multiprocessing

import time

params = Params("./params/xadd_params_reservoir.json")

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
print("Context 1 Nodes: ", len(str(reward_node_1)))
print(reward_node_1)
context_1.save_graph(model_1.reward, '1')

reward_node_2 = context_2._id_to_node.get(model_2.reward)
print("Reward XADD 2")
print("Context 2 Nodes: ", len(str(reward_node_2)))
print(reward_node_2)
context_2.save_graph(model_2.reward,'./2')

reward_node_diff = context_diff._id_to_node.get(model_diff.reward)
print("Reward XADD Diff")
print("Context diff Nodes: ", len(str(reward_node_diff)))
print(reward_node_diff)
context_diff.save_graph(model_diff.reward,'./diff')


def gen_xadd_files(horizon_length):
  print('{} xadd horizon is porcessing.......'.format(horizon_length))

  start_time = time.time()
  vid_1, q_1 = ModelDiff.do_SDP(model_1, context_1, params.sdp_mode, params.discount_rate, horizon_length)
  end_time = time.time()
  print('SDP 1',start_time - end_time)


  start_time = time.time()
  vid_2, q_2 = ModelDiff.do_SDP(model_2, context_2, params.sdp_mode, params.discount_rate, horizon_length)
  end_time = time.time()
  print('SDP 2',start_time - end_time)

  start_time = time.time()
  vid_diff, q_diff = ModelDiff.do_SDP(model_diff, context_diff, params.sdp_mode, params.discount_rate, horizon_length)
  end_time = time.time()
  print('SDP diff',start_time - end_time)


  print("Context 1 Nodes: ", len(context_1._id_to_node))
  print("Context 2 Nodes: ",len(context_2._id_to_node))
  print("Context diff Nodes: ",len(context_diff._id_to_node))



  if params.save_xadds:
    xadd_path = pathlib.Path(params.save_path+'{}_step/'.format(horizon_length))
    xadd_path.mkdir(parents=True, exist_ok=True)

    save_value_function(params.save_path+'{}_step/'.format(horizon_length), 'v_source', vid_1, context_1)
    save_value_function(params.save_path+'{}_step/'.format(horizon_length), 'v_target', vid_2, context_2)
    save_value_function(params.save_path+'{}_step/'.format(horizon_length), 'v_diff', vid_diff, context_diff)

    save_q_function(params.save_path+'{}_step/'.format(horizon_length), 'q_source', q_1, context_1)
    save_q_function(params.save_path+'{}_step/'.format(horizon_length), 'q_target', q_2, context_2)
    save_q_function(params.save_path+'{}_step/'.format(horizon_length), 'q_diff', q_diff, context_diff)

  print('{} xadd horizon completed.'.format(horizon_length))
  

def main():

  gen_xadd_files(params.horizon_length)

  # num_processes = multiprocessing.cpu_count()
  # pool = multiprocessing.Pool(processes=num_processes)

  # results = pool.map(gen_xadd_files, params.horizon_length)

  # pool.close()

  # pool.join()

  # print('all jobs completed')

main()
     



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
## TEST SINGLE VALUE


###############################################

# test_dict_c_1 = {'stock___i1':0}
# test_dict_c_2 = {'stock___i1':0}
# test_dict_c_diff = {'stock___i1':0}
# test_dict_b = {}

# result_list_1 = []
# result_list_2 = []
# result_list_diff = []

# q_1_v_0_lst = []
# q_1_v_1_lst = []
# q_2_v_0_lst = []
# q_2_v_1_lst = []
# v_1_v_lst = []
# v_2_v_lst = []

# for i in range(0, 51, 1):
#     test_dict_c_1['stock___i1'] = i
#     test_dict_c_2['stock___i1'] = i
#     test_dict_c_diff['stock___i1'] = i
#     q_1_v_0 = ModelDiff.eval_function(test_dict_b, test_dict_c_1, q_1[0][1], model_1, context_1)
#     q_1_v_1 = ModelDiff.eval_function(test_dict_b, test_dict_c_1, q_1[1][1], model_1, context_1)
#     q_2_v_0 = ModelDiff.eval_function(test_dict_b, test_dict_c_2, q_2[0][1], model_2, context_2)
#     q_2_v_1 = ModelDiff.eval_function(test_dict_b, test_dict_c_2, q_2[1][1], model_2, context_2)
#     v_1_v = ModelDiff.eval_function(test_dict_b, test_dict_c_1, vid_1, model_1, context_1)
#     v_2_v = ModelDiff.eval_function(test_dict_b, test_dict_c_2, vid_2, model_1, context_2)
#     q_1_v_0_lst.append(q_1_v_0)
#     q_1_v_1_lst.append(q_1_v_1)
#     q_2_v_0_lst.append(q_2_v_0)
#     q_2_v_1_lst.append(q_2_v_1)
#     v_1_v_lst.append(v_1_v)
#     v_2_v_lst.append(v_2_v)

#     # print(i, v_1_v, v_2_v, (q_1_v_0 > q_1_v_1) == (q_2_v_0 > q_2_v_1))
#     print(i, q_1_v_0, q_1_v_1, 'x', q_2_v_0,  q_2_v_1)
#     # print(i, v_1_v, v_2_v)

    # q_diff_v = ModelDiff.eval_function(test_dict_b, test_dict_c_diff, q_diff[0][1], model_diff, context_diff)
    # v_1_v = ModelDiff.eval_function(test_dict_b, test_dict_c_1, vid_1, model_1, context_1)
    # v_2_v = ModelDiff.eval_function(test_dict_b, test_dict_c_2, vid_2, model_1, context_2)
    # print(i, v_1_v, v_2_v)
    # print(i, q_1_v, q_diff_v, q_2_v, q_1_v + q_diff_v - q_2_v, q_1_v + q_diff_v == q_2_v)
  
# x = list(range(0, len(v_1_v_lst)))
# # plt.plot(x, v_1_v_lst, label = "v_source")
# # plt.plot(x, v_2_v_lst, label = "v_target")

# plt.plot(x, q_1_v_0_lst, label = "q_source_purchase_true")
# plt.plot(x, q_1_v_1_lst, label = "q_source_purchase_false")

# plt.plot(x, q_2_v_0_lst, label = "q_target_purchase_true")
# plt.plot(x, q_2_v_1_lst, label = "q_target_purchase_false")


# # plt.plot(x, res1_ppr_list, label = "ppr")
# # plt.plot(x, res1_rs_list, label = "reward shaping")
# plt.title("holding source 2 target 0")
# plt.xlabel("stock_level")
# plt.ylabel("q-values")
# plt.legend()
# # plt.show()
# plt.savefig("test_out")



# test_dict_c_1 = {'rlevel___t1':0}
# test_dict_c_2 = {'rlevel___t1':0}
# test_dict_c_diff = {'rlevel___t1':0}
# test_dict_b = {}

# result_list_1 = []
# result_list_2 = []
# result_list_diff = []

# for i in range(0, 51, 1):
#     test_dict_c_1['rlevel___t1'] = i
#     test_dict_c_2['rlevel___t1'] = i
#     test_dict_c_diff['rlevel___t1'] = i
#     q_1_v_0 = ModelDiff.eval_function(test_dict_b, test_dict_c_1, q_1[0][1], model_1, context_1)
#     q_1_v_1 = ModelDiff.eval_function(test_dict_b, test_dict_c_1, q_1[1][1], model_1, context_1)
#     q_2_v_0 = ModelDiff.eval_function(test_dict_b, test_dict_c_2, q_2[0][1], model_2, context_2)
#     q_2_v_1 = ModelDiff.eval_function(test_dict_b, test_dict_c_2, q_2[1][1], model_2, context_2)
#     v_1_v = ModelDiff.eval_function(test_dict_b, test_dict_c_1, vid_1, model_1, context_1)
#     v_2_v = ModelDiff.eval_function(test_dict_b, test_dict_c_2, vid_2, model_1, context_2)
#     print(i, (q_1_v_0 >= q_1_v_1) == (q_2_v_0 >= q_2_v_1))
#     # print(i, q_1_v_0, q_1_v_1, 'x', q_2_v_0,  q_2_v_1)
#     # print(i, v_1_v, v_2_v)

#     # q_diff_v = ModelDiff.eval_function(test_dict_b, test_dict_c_diff, q_diff[0][1], model_diff, context_diff)
#     # v_1_v = ModelDiff.eval_function(test_dict_b, test_dict_c_1, vid_1, model_1, context_1)
#     # v_2_v = ModelDiff.eval_function(test_dict_b, test_dict_c_2, vid_2, model_1, context_2)
#     # print(i, v_1_v, v_2_v)
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


# test_dict_c_1 = {'rlevel___t1':0,'rlevel___t2':0}
# test_dict_c_2 = {'rlevel___t1':0,'rlevel___t2':0}
# test_dict_c_diff = {'rlevel___t1':0,'rlevel___t2':0}
# test_dict_b = {}

# result_list_1 = []
# result_list_2 = []
# result_list_diff = []

# max_error = 0

# for n in range(4):
#     for i in range(0, 51, 5):
#         for j in range(0, 51, 5):
#             test_dict_c_1['rlevel___t1'] = i
#             test_dict_c_1['rlevel___t2'] = j
#             test_dict_c_2['rlevel___t1'] = i
#             test_dict_c_2['rlevel___t2'] = j
#             test_dict_c_diff['rlevel___t1'] = i
#             test_dict_c_diff['rlevel___t2'] = j
#             q_1_v_0 = ModelDiff.eval_function(test_dict_b, test_dict_c_1, q_1[n][0], model_1, context_1)
#             q_2_v_0 = ModelDiff.eval_function(test_dict_b, test_dict_c_2, q_2[n][0], model_2, context_2)
#             q_diff_v_0 = ModelDiff.eval_function(test_dict_b, test_dict_c_diff, q_diff[n][0], model_diff, context_diff)
#             q_1_v_1 = ModelDiff.eval_function(test_dict_b, test_dict_c_1, q_1[n][1], model_1, context_1)
#             q_2_v_1 = ModelDiff.eval_function(test_dict_b, test_dict_c_2, q_2[n][1], model_2, context_2)
#             q_diff_v_1 = ModelDiff.eval_function(test_dict_b, test_dict_c_diff, q_diff[n][1], model_diff, context_diff)
#             # print(i,j, q_1_v, q_2_v, q_diff_v, q_1_v + q_diff_v,  q_1_v + q_diff_v - q_2_v, q_1_v + q_diff_v == q_2_v)
#             # if max_error < abs(q_1_v + q_diff_v - q_2_v):
#             #     max_error = abs(q_1_v + q_diff_v - q_2_v)
#             print(i,j, (q_1_v_0 >= q_1_v_1) == (q_2_v_0 >= q_2_v_1))

# print(max_error)