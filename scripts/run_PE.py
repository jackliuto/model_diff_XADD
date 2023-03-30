from modeldiff.diffenv.diffClass import ModelDiffNavigation


# ## experiment 1
# policy_path = '../policies/op_55/'
# m1_path = "../RDDL/Navigation_disc_goal_551010/domain.rddl"
# m2_path = "../RDDL/Navigation_disc_goal_771010/domain.rddl"
# title_1 = 'Value at T=2 for goal=(5,5)'
# title_2 = 'Value at T=2 for goal=(7,7)'
# title_diff = 'Value at T=2 for goal=(7,7) - goal=(5,5)'
# fpath_1_value = '../visualization/value_maps/vplot551010.png'
# fpath_2_value = '../visualization/value_maps/vplot771010.png'
# fpath_diff_value = '../visualization/value_maps/vplot_diff_551010771010.png'
# fpath_1_xadd_reward = '../visualization/xadd_graphs/xadd_reward_551010.png'
# fpath_2_xadd_reward = '../visualization/xadd_graphs/xadd_reward_771010.png'
# fpath_diff_xadd_reward = '../visualization/xadd_graphs/xadd_reward_diff_551010771010.png'
# fpath_1_xadd_value = '../visualization/xadd_graphs/xadd_value_551010.png'
# fpath_2_xadd_value = '../visualization/xadd_graphs/xadd_value_771010.png'
# fpath_diff_xadd_value = '../visualization/xadd_graphs/xadd_value__diff_551010771010.png'

# # experiment 2
# policy_path = '../policies/op_50/'
# m1_path = "../RDDL/Navigation_disc_goal_501010/domain.rddl"
# m2_path = "../RDDL/Navigation_disc_goal_551010/domain.rddl"
# title_1 = 'Value at T=2 for goal=(5,0)'
# title_2 = 'Value at T=2 for goal=(5,5)'
# title_diff = 'Value at T=2 for goal=(5,5) - goal=(5,0)'
# fpath_1_value = '../visualization/value_maps/vplot501010.png'
# fpath_2_value = '../visualization/value_maps/vplot551010.png'
# fpath_diff_value = '../visualization/value_maps/vplot_diff_501010551010.png'
# fpath_1_xadd_reward = '../visualization/xadd_graphs/xadd_reward_501010.png'
# fpath_2_xadd_reward = '../visualization/xadd_graphs/xadd_reward_551010.png'
# fpath_diff_xadd_reward = '../visualization/xadd_graphs/xadd_reward_diff_501010551010.png'
# fpath_1_xadd_value = '../visualization/xadd_graphs/xadd_value_501010.png'
# fpath_2_xadd_value = '../visualization/xadd_graphs/xadd_value_551010.png'
# fpath_diff_xadd_value = '../visualization/xadd_graphs/xadd_value__diff_501010551010.png'


# experiment 3
policy_path = '../policies/op_77/'
m1_path = "../RDDL/Navigation_disc_goal_771010_55s/domain.rddl"
m2_path = "../RDDL/Navigation_disc_goal_0033_55s/domain.rddl"
title_1 = 'Value at T=2 for goal=(7,7)'
title_2 = 'Value at T=2 for goal=(3,3)'
title_diff = 'Value at T=2 for goal=(3,3) - goal=(7,7)'
fpath_1_value = '../visualization/value_maps/vplot771010_55s.png'
fpath_2_value = '../visualization/value_maps/vplot0033_55s.png'
fpath_diff_value = '../visualization/value_maps/vplot_diff_7710100033_55s.png'
fpath_1_xadd_reward = '../visualization/xadd_graphs/xadd_reward_771010_55s.png'
fpath_2_xadd_reward = '../visualization/xadd_graphs/xadd_reward_0033_55s.png'
fpath_diff_xadd_reward = '../visualization/xadd_graphs/xadd_reward_diff_7710100033_55s.png'
fpath_1_xadd_value = '../visualization/xadd_graphs/xadd_value_771010_55s.png'
fpath_2_xadd_value = '../visualization/xadd_graphs/xadd_value_0033_55s.png'
fpath_diff_xadd_value = '../visualization/xadd_graphs/xadd_value__diff_7710100033_55s.png'


ModelDiff = ModelDiffNavigation(policy_path,m1_path, m2_path)
ModelDiff.build_model_with_diff_reward()

model_1 = ModelDiff._model_1
context_1 = ModelDiff._context_1

model_2 = ModelDiff._model_2
context_2 = ModelDiff._context_2

model_diff = ModelDiff._model_diff
context_diff = ModelDiff._context_diff


vid_1 = ModelDiff.do_PE(model_1, context_1, 2)
vid_2 = ModelDiff.do_PE(model_2, context_2, 2)
vid_diff = ModelDiff.do_PE(model_diff, context_diff, 2)

ModelDiff.gen_value_graph(title_1, fpath_1_value, model_1, context_1, vid_1)
ModelDiff.gen_value_graph(title_2, fpath_2_value, model_2, context_2, vid_2)
ModelDiff.gen_value_graph(title_diff, fpath_diff_value, model_diff, context_diff, vid_diff)

context_1.save_graph_png(model_1.reward, fpath_1_xadd_reward)
context_2.save_graph_png(model_2.reward, fpath_2_xadd_reward)
context_diff.save_graph_png(model_diff.reward, fpath_diff_xadd_reward)

context_1.save_graph_png(vid_1, fpath_1_xadd_value)
context_2.save_graph_png(vid_2, fpath_2_xadd_value)
context_diff.save_graph_png(vid_diff, fpath_diff_xadd_value)