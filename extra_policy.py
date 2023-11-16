import warnings
warnings.filterwarnings("ignore")

from sklearn import tree

from matplotlib import pyplot as plt
import numpy as np

from pyRDDLGym import RDDLEnv
from policy_learning.envs.navEnv import envWrapperNav
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import DQN

# globalvars
DOMAIN_PATH = "./RDDL/navigation/navigation_disc/domain.rddl"
INSTANCE_PATH = "./RDDL/navigation/navigation_disc/instance_1agent_source.rddl"



RDDLEnv = RDDLEnv.RDDLEnv(domain=DOMAIN_PATH, instance=INSTANCE_PATH)
env = envWrapperNav(RDDLEnv, max_episode_length=1000)


loaded_model = DQN.load('../checkpoints/rl_model_9000000_steps.zip', env=env, exploration_fraction=0.0)

obs, _ = env.reset()

q_values = loaded_model.predict(obs)

x = np.arange(0, 11, 1)
y = np.arange(0, 11, 1)

X, Y = np.meshgrid(x, y)
Z = np.zeros_like(X, dtype=float)


action_list = env.action_list
observation_list = env.observation_list

inputs = []
labels = []

for i in range(len(x)):
    for j in range(len(y)):
        obs = {'pos_x___a1': np.array([i], dtype=np.float32), 'pos_y___a1': np.array([j], dtype=np.float32)}
        value, _ = loaded_model.predict(obs)
        Z[i][j] = value
        inputs.append(np.array([i,j]))
        labels.append(env.action_list[value])


inputs = np.array(inputs)
labels = np.array(labels)


clf = tree.DecisionTreeClassifier()
clf = clf.fit(inputs, labels)

fig = plt.figure(figsize=(50, 30))
tree.plot_tree(clf)
fig.savefig('test_tree.png')

print(Z.T)

y_pred = clf.predict(inputs)

Z_tree = np.empty(X.shape, dtype=float)

for idx in range(len(inputs)):
    i = inputs[idx][0]
    j = inputs[idx][1]
    value = y_pred[idx]
    Z_tree[i][j] = action_list.index(value)


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
                "value": tree.threshold[i],
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

print(clf.tree_.feature)

print(export_dict(clf, observation_list))




# # tree_dict = tree_to_dict(clf, tree_rules)
# tree_dict = tree_text_to_dict(tree_rules)

# print(tree_rules)

