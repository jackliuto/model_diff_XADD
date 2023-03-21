import pyRDDLGym
from pyRDDLGym import RDDLEnv
from graph import RDDL2GraphFile







# domain_name = 'Navigation_disc'
# domain_path = '../RDDL/Navigation_disc/domain.rddl'
# instance_path = '../RDDL/Navigation_disc/instance0.rddl'

# domain_name = 'Navigation_disc2'
# domain_path = '../RDDL/Navigation_disc2/domain.rddl'
# instance_path = '../RDDL/Navigation_disc2/instance0.rddl'

# domain_name = 'Navigation_disc_linear'
# domain_path = '../RDDL/Navigation_disc_linear/domain.rddl'
# instance_path = '../RDDL/Navigation_disc_linear/instance0.rddl'

# domain_name = 'Navigation_disc_linear2'
# domain_path = '../RDDL/Navigation_disc_linear2/domain.rddl'
# instance_path = '../RDDL/Navigation_disc_linear2/instance0.rddl'

# domain_name = 'Navigation_disc_linear_policy'
# domain_path = '../RDDL/Navigation_disc_linear_policy/domain.rddl'
# instance_path = '../RDDL/Navigation_disc_linear_policy/instance0.rddl'

# domain_name = 'Navigation_cont_error'
# domain_path = '../RDDL/Navigation_cont_error/domain.rddl'
# instance_path = '../RDDL/Navigation_cont_error/instance0.rddl'

# domain_name = 'Navigation_cont'
# domain_path = '../RDDL/Navigation_cont/domain.rddl'
# instance_path = '../RDDL/Navigation_cont/instance0.rddl'

# domain_name = 'Navigation_cont2'
# domain_path = '../RDDL/Navigation_cont2/domain.rddl'
# instance_path = '../RDDL/Navigation_cont2/instance0.rddl'

# domain_name = 'Navigation_cont_hole'
# domain_path = '../RDDL/Navigation_cont_hole/domain.rddl'
# instance_path = '../RDDL/Navigation_cont_hole/instance0.rddl'

# domain_name = 'Navigation_cont_border'
# domain_path = '../RDDL/Navigation_cont_border/domain.rddl'
# instance_path = '../RDDL/Navigation_cont_border/instance0.rddl'

# domain_name = 'PowerGen_linear'
# domain_path = '../RDDL/PowerGen_linear/domain.rddl'
# instance_path = '../RDDL/PowerGen_linear/instance0.rddl'


domain_name = 'Navigation_disc_goal'
domain_path = '../RDDL/Navigation_disc_goal/domain.rddl'
instance_path = '../RDDL/Navigation_disc_goal/instance0.rddl'

# domain_name_new = 'Navigation_disc_linear_goal2'
# domain_path_new = '../RDDL/Navigation_disc_linear_goal2/domain.rddl'
# instance_path_new = '../RDDL/Navigation_disc_linear_goal/instance0.rddl'

myEnv = RDDLEnv.RDDLEnv(domain=domain_path, instance=instance_path)

r2g = RDDL2GraphFile(
    domain_name=domain_name, instance_number=0,
    domain_path=domain_path, instance_path=instance_path,
    directed=True,
    strict_grouping=True
)

r2g.save_dbn(file_name=domain_name)






