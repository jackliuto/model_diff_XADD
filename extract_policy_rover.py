import warnings
warnings.filterwarnings("ignore")

from sklearn import tree

from matplotlib import pyplot as plt
import numpy as np

from pyRDDLGym import RDDLEnv
from policy_learning.envs.roverEnv import envWrapperRover
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import DQN

from xaddpy.xadd.xadd import XADD

import json

import pdb

# globalvars
DOMAIN_PATH = "./RDDL/mars_rover/domain.rddl"
INSTANCE_PATH = "./RDDL/mars_rover/instance.rddl"



RDDLEnv = RDDLEnv.RDDLEnv(domain=DOMAIN_PATH, instance=INSTANCE_PATH)
env = envWrapperRover(RDDLEnv, max_episode_length=1000)

loaded_model = DQN.load('../checkpoints/mars_rover/rl_model_8368000_steps.zip', env=env, exploration_fraction=0.0)

obs, _ = env.reset()

q_values = loaded_model.predict(obs)

x = np.arange(0, 11, 1)
y = np.arange(0, 11, 1)

X, Y = np.meshgrid(x, y)



action_list = env.action_list
observation_list = env.observation_list

print(action_list)

inputs = []
labels = []

Z = np.zeros_like(X, dtype=float)

for i in range(len(x)):
    for j in range(len(y)):
        obs = {'pos_x___a1': np.array([i], dtype=np.float32), 'pos_y___a1': np.array([j], dtype=np.float32), 'has_mineral___a1':0}
        value, _ = loaded_model.predict(obs)
        Z[i][j] = value
        inputs.append(np.array([i,j]))
        labels.append(env.action_list[value])


print(Z.T)


Z = np.zeros_like(X, dtype=float)

for i in range(len(x)):
    for j in range(len(y)):
        obs = {'pos_x___a1': np.array([i], dtype=np.float32), 'pos_y___a1': np.array([j], dtype=np.float32), 'has_mineral___a1':1}
        value, _ = loaded_model.predict(obs)
        Z[i][j] = value
        inputs.append(np.array([i,j]))
        labels.append(env.action_list[value])

print(Z.T)


inputs = np.array(inputs)
labels = np.array(labels)


clf = tree.DecisionTreeClassifier(max_depth=None)
clf = clf.fit(inputs, labels)

fig = plt.figure(figsize=(50, 30))
tree.plot_tree(clf)
fig.savefig('test_tree.png')

y_pred = clf.predict(inputs)

Z_tree_0 = np.empty(X.shape, dtype=float)

for idx in range(len(inputs)):
    i = inputs[idx][0]
    j = inputs[idx][1]
    value = y_pred[idx]
    Z_tree[i][j] = action_list.index(value)

print(Z_tree.T)

def export_dict(clf, feature_names=None):
    tree = clf.tree_
    if feature_names is None:
        feature_names = range(clf.max_features_)
    
    # Build tree nodes
    tree_nodes = []
    for i in range(tree.node_count):
        if (tree.children_left[i] == tree.children_right[i]):
            tree_nodes.append(
                clf.classes_[np.argmax(tree.value[i])]
            )
        else:
            tree_nodes.append({
                "feature": feature_names[tree.feature[i]],
                "threshold": tree.threshold[i],
                "left": tree.children_left[i],
                "right": tree.children_right[i],
            })
    
    # Link tree nodes
    for node in tree_nodes:
        if isinstance(node, dict):
            node["left"] = tree_nodes[node["left"]]
        if isinstance(node, dict):
            node["right"] = tree_nodes[node["right"]]
    
    # Return root node
    return tree_nodes[0]

policy_dict = export_dict(clf, observation_list)

print(policy_dict)


def policy_dict2xadd_str(node):
    # Base case: if the node is a leaf (i.e., a string), return it directly
    if isinstance(node, str):
        return f"([{node}])"

    # Recursive case: process left and right children
    left_str = policy_dict2xadd_str(node['left']) if 'left' in node else ''
    right_str = policy_dict2xadd_str(node['right']) if 'right' in node else ''

    # Format the current node string
    node_str = f"( [{node['feature']} <= {node['threshold']}] {left_str} {right_str} )"

    return node_str

policy_xadd_str = policy_dict2xadd_str(policy_dict)

print(policy_xadd_str)

xadd = XADD()
policy_id = xadd.import_xadd(xadd_str=policy_xadd_str)
policy_id = xadd.reduce_lp(policy_id)

print(policy_id)

print(xadd.get_repr(policy_id))

def gen_policy_dict(action_list, all_policy_id, xadd):
    print(xadd._str_var_to_var)
    policy_dict = {}
    for action in action_list:
        if action in xadd._str_var_to_var.keys():
            sub_dict = {}
            for a in action_list:
                if a in xadd._str_var_to_var.keys():
                    if a == action:
                        sub_dict[xadd._str_var_to_var[a]] = 1
                    else:
                        sub_dict[xadd._str_var_to_var[a]] = 0

            policy_id = xadd.substitute(all_policy_id, sub_dict)



            policy_id = xadd.reduce_lp(policy_id)
            policy_id = xadd.unary_op(policy_id, 'float')
            policy_node = xadd._id_to_node[policy_id]
            policy_node.turn_off_print_node_info()
            policy_dict[action] = str(policy_node)
            policy_node.turn_on_print_node_info()
        else:
            policy_dict[action] = "([0])"
        
    return policy_dict

policy_dict = gen_policy_dict(action_list, policy_id, xadd)

for k,v in policy_dict.items():
    print(k)
    print(v)

json.dump(policy_dict, open('./exported_policy/navigation/policy_dict.json', 'w'))

def plot_value(policy_dict, x_range, y_range):

    x = np.arange(x_range[0], x_range[1], 1)
    y = np.arange(y_range[0], y_range[1], 1)
    X, _ = np.meshgrid(x, y)

    Z_all = np.zeros_like(X, dtype=float)

    for k, v in policy_dict.items():
        print('------------------')
        print(k)
        print(v)

        context = XADD()

        value_id = context.import_xadd(xadd_str=v)

        # x = np.arange(x_range[0], x_range[1], 1)
        # y = np.arange(y_range[0], y_range[1], 1)

        # X, Y = np.meshgrid(x, y)
        Z = np.zeros_like(X, dtype=float)

        var_set = context.collect_vars(value_id)

        var_dict = {}
        for i in var_set:
            var_dict[f"{i}"] = i

        for i in range(len(x)):
            for j in range(len(y)):
                cont_assign = {var_dict["pos_x___a1"]: j,
                        var_dict["pos_y___a1"]: i,}
                bool_assign = {}
                value = context.evaluate(value_id, bool_assign=bool_assign, cont_assign=cont_assign)
                Z[i][j] = value

        print(k)
        print(Z.T)
        Z_all += Z.T
    print(Z_all)

    # fig = plt.figure(figsize=(50, 30))
    # plt.imshow(Z.T, origin='lower', interpolation='none')
    # plt.colorbar()
    # plt.show()

plot_value(policy_dict, [0,11], [0,11])

