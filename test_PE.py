from modeldiff.diffenv.diffClass import ModelDiffReservoir

## experiment 1
policy_path = 'policies/res/'
m1_path = "RDDL/Reservoir_disc_simple_1/domain.rddl"
m2_path = "RDDL/Reservoir_disc_simple_2/domain.rddl"

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
vid_1 = ModelDiff.do_PE(model_1, context_1, 5)
print('MDP2')
vid_2 = ModelDiff.do_PE(model_2, context_2, 5)
print('MDP_diff')
vid_diff = ModelDiff.do_PE(model_diff, context_diff, )

print(context_1._id_to_node.get(vid_1))
print(context_2._id_to_node.get(vid_2))
print(context_diff._id_to_node.get(vid_diff))

test_dict_c_1 = {'rlevel___t1':0,'rlevel___t2':0}
test_dict_c_2 = {'rlevel___t1':0,'rlevel___t2':0}
test_dict_c_diff = {'rlevel___t1':0,'rlevel___t2':0}
test_dict_b = {}

result_list_1 = []
result_list_2 = []
result_list_diff = []
for i in range(0, 101, 5):
    for j in range(0, 101, 5):
        test_dict_c_1['rlevel___t1'] = i
        test_dict_c_1['rlevel___t2'] = j
        test_dict_c_2['rlevel___t1'] = i
        test_dict_c_2['rlevel___t2'] = j
        test_dict_c_diff['rlevel___t1'] = i
        test_dict_c_diff['rlevel___t2'] = j
        v_1 = ModelDiff.eval_function(test_dict_b, test_dict_c_1, vid_1, model_1, context_1)
        v_2 = ModelDiff.eval_function(test_dict_b, test_dict_c_2, vid_2, model_2, context_2)
        v_diff = ModelDiff.eval_function(test_dict_b, test_dict_c_diff, vid_diff, model_diff, context_diff)
        print(i, v_1, v_2, v_diff, v_1 + v_diff == v_2)

