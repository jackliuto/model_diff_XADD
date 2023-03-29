from modeldiff.diffenv.diffClass import ModelDiff

m1_path = "../RDDL/Navigation_disc_goal_551010/domain.rddl"
m2_path = "../RDDL/Navigation_disc_goal_771010/domain.rddl"

model_diff = ModelDiff(m1_path, m2_path)
model_diff.build_model_with_diff_reward()