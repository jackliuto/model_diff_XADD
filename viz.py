import matplotlib.pyplot as plt
import numpy as np
import random

import json




# res1_dqn_dict = json.load(open('./results/reservoir_2res_20steps_dqn.json'))
# res1_ppr_dict = json.load(open('./results/reservoir_2res_20steps_ppr.json'))
# res1_rs_dict = json.load(open('./results/reservoir_2res_20steps_rewardshaping.json'))
# res1_lb_dict = json.load(open('./results/reservoir_2res_20steps_lowerbound.json'))
# res1_cf_dict = json.load(open('./results/reservoir_2res_20steps_lowerbound_counterfactual.json'))


res1_dqn_dict = json.load(open('./results/navigation/1agent_10steps_10eps_dqn.json'))
# res1_ppr_dict = json.load(open('./results/navigation/reservoir_2res_20steps_ppr.json'))
res1_rs_dict = json.load(open('./results/navigation/1agent_10steps_10eps_rewardshaping.json'))
res1_lb_dict = json.load(open('./results/navigation/1agent_10steps_10eps_lowerbound.json'))
# res1_cf_dict = json.load(open('./results/navigation/reservoir_2res_20steps_lowerbound_counterfactual.json'))


# res1_dqn_dict = json.load(open('./results/navigation_1agent_20steps_dqn.json'))
# res1_ppr_dict = json.load(open('./results/navigation_1agent_20steps_ppr.json'))
# res1_rs_dict = json.load(open('./results/navigation_1agent_20steps_rewardshaping.json'))
# res1_lb_dict = json.load(open('./results/navigation_1agent_20steps_lowerbound.json'))
# res1_cf_dict = json.load(open('./results/navigation_1agent_20steps_lowerbound_counterfactual.json'))


# res1_lb_list = [sum(res1_lb_dict['eval_reward'][:i])/(i+1) for i, v in enumerate(res1_lb_dict['eval_reward'])]
# res1_dqn_list = [sum(res1_dqn_dict['eval_reward'][:i])/(i+1) for i, v in enumerate(res1_dqn_dict['eval_reward'])]
# res1_ppr_list = [sum(res1_ppr_dict['eval_reward'][:i])/(i+1) for i, v in enumerate(res1_ppr_dict['eval_reward'])]
# res1_rs_list = [sum(res1_rs_dict['eval_reward'][:i])/(i+1) for i, v in enumerate(res1_rs_dict['eval_reward'])]

avg_dqn_list = np.mean(res1_dqn_dict['eval_reward'], axis=0)
avg_lb_list = np.mean(res1_lb_dict['eval_reward'], axis=0)
# avg_ppr_list = np.mean(res1_ppr_dict['eval_reward'], axis=0)
avg_rs_list = np.mean(res1_rs_dict['eval_reward'], axis=0)
# avg_cf_list = np.mean(res1_cf_dict['eval_reward'], axis=0)


# res1_lb_list = [res1_lb_dict['eval_reward'][i] for i, v in enumerate(res1_lb_dict['eval_reward'])]
# res1_dqn_list = [res1_dqn_dict['eval_reward'][i] for i, v in enumerate(res1_dqn_dict['eval_reward'])]
# res1_ppr_list = [res1_ppr_dict['eval_reward'][i] for i, v in enumerate(res1_ppr_dict['eval_reward'])]
# res1_rs_list = [res1_rs_dict['eval_reward'][i] for i, v in enumerate(res1_rs_dict['eval_reward'])]

def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / float(N)

res1_lb_list = running_mean(avg_lb_list, 10)
res1_dqn_list = running_mean(avg_dqn_list, 10)
# res1_ppr_list = running_mean(avg_ppr_list, 50)
res1_rs_list = running_mean(avg_rs_list, 10)
# res1_cf_list = running_mean(avg_cf_list, 50)

x = list(range(0, len(res1_dqn_list)))

plt.plot(x, res1_lb_list, label = "lowerbound")
plt.plot(x, res1_dqn_list, label = "DQN",  linestyle='-.')
# plt.plot(x, res1_ppr_list, label = "ppr",  linestyle='-.')
plt.plot(x, res1_rs_list, label = "reward shaping",  linestyle='-.')
# plt.plot(x, res1_cf_list, label = "Lowerbound CF")
plt.title("Reservoir")
plt.xlabel("epsisode")
plt.ylabel("Mean Discounted Reward")
plt.legend()
plt.show()
# plt.savefig("Reservoir 2Res 10")


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