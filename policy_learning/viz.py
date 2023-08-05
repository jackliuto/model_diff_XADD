import matplotlib.pyplot as plt
import numpy as np

import json

res1_lb_dict = json.load(open('/home/jackliu/model-diff/model_diff_DQN/results/1res_DQN_lowerbound.json'))
res1_dqn_dict = json.load(open('/home/jackliu/model-diff/model_diff_DQN/results/1res_DQN_vanilla.json'))
res1_ppr_dict = json.load(open('/home/jackliu/model-diff/model_diff_DQN/results/1res_DQN_ppr.json'))

res1_lb_list = [sum(res1_lb_dict['eval_reward'][:i])/(i+1) for i, v in enumerate(res1_lb_dict['eval_reward'])]
res1_dqn_list = [sum(res1_dqn_dict['eval_reward'][:i])/(i+1) for i, v in enumerate(res1_dqn_dict['eval_reward'])]
res1_ppr_list = [sum(res1_ppr_dict['eval_reward'][:i])/(i+1) for i, v in enumerate(res1_ppr_dict['eval_reward'])]


x = list(range(0, len(res1_ppr_list)))

plt.plot(x, res1_lb_list, label = "LB")
plt.plot(x, res1_dqn_list, label = "DQN")
plt.plot(x, res1_ppr_list, label = "ppr")
plt.legend()
plt.show()


# res2_lb_dict = json.load(open('/home/jackliu/model-diff/model_diff_DQN/results/2res_DQN_lowerbound.json'))
# res2_dqn_dict = json.load(open('/home/jackliu/model-diff/model_diff_DQN/results/2res_DQN_vanilla.json'))
# res2_ppr_dict = json.load(open('/home/jackliu/model-diff/model_diff_DQN/results/2res_DQN_ppr.json'))

# res2_lb_list = [sum(res2_lb_dict['eval_reward'][:i])/(i+1) for i, v in enumerate(res2_lb_dict['eval_reward'])]
# res2_dqn_list = [sum(res2_dqn_dict['eval_reward'][:i])/(i+1) for i, v in enumerate(res2_dqn_dict['eval_reward'])]
# res2_ppr_list = [sum(res2_ppr_dict['eval_reward'][:i])/(i+1) for i, v in enumerate(res2_ppr_dict['eval_reward'])]


# x = list(range(0, len(res2_ppr_list)))

# plt.plot(x, res2_lb_list, label = "LB")
# plt.plot(x, res2_dqn_list, label = "DQN")
# plt.plot(x, res2_ppr_list, label = "ppr")
# plt.legend()
# plt.show()