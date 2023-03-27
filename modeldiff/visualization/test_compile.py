from pyRDDLGym import RDDLEnv

from pyRDDLGym.XADD.RDDLModelXADD import RDDLModelWXADD
# domain_name = 'PowerGen_linear'
# domain_path = '../RDDL/PowerGen_linear/domain.rddl'
# instance_path = '../RDDL/PowerGen_linear/instance0.rddl'

# domain_name_new = 'PowerGen_linear_higherDemandPen'
# domain_path_new = '../RDDL/PowerGen_linear_higherDemandPen/domain.rddl'
# instance_path_new = '../RDDL/PowerGen_linear_higherDemandPen/instance0.rddl'

# domain_name = 'Navigation_cont'
# domain_path = '../RDDL/Navigation_cont/domain.rddl'
# instance_path = '../RDDL/Navigation_cont/instance0.rddl'

# domain_name_new = 'Navigation_cont_hole'
# domain_path_new = '../RDDL/Navigation_cont_hole/domain.rddl'
# instance_path_new = '../RDDL/Navigation_cont_hole/instance0.rddl'

# domain_name = 'Navigation_cont'
# domain_path = '../RDDL/Navigation_cont/domain.rddl'
# instance_path = '../RDDL/Navigation_cont/instance0.rddl'

# domain_name_new = 'Navigation_cont_border'
# domain_path_new = '../RDDL/Navigation_cont_border/domain.rddl'
# instance_path_new = '../RDDL/Navigation_cont_border/instance0.rddl'

# domain_name = 'Navigation_disc'
# domain_path = '../RDDL/Navigation_disc/domain.rddl'
# instance_path = '../RDDL/Navigation_disc/instance0.rddl'

# domain_name_new = 'Navigation_disc2'
# domain_path_new = '../RDDL/Navigation_disc2/domain.rddl'
# instance_path_new = '../RDDL/Navigation_disc2/instance0.rddl'

# domain_name = 'Navigation_disc_linear'
# domain_path = '../RDDL/Navigation_disc_linear/domain.rddl'
# instance_path = '../RDDL/Navigation_disc/instance0.rddl'

# domain_name_new = 'Navigation_disc_linear2'
# domain_path_new = '../RDDL/Navigation_disc_linear2/domain.rddl'
# instance_path_new = '../RDDL/Navigation_disc_linear2/instance0.rddl'

domain_name = 'Navigation_disc_linear'
domain_path = '../RDDL/Navigation_disc_linear/domain.rddl'
instance_path = '../RDDL/Navigation_disc/instance0.rddl'

domain_name_new = 'Navigation_disc_linear_policy'
domain_path_new = '../RDDL/Navigation_disc_linear_policy/domain.rddl'
instance_path_new = '../RDDL/Navigation_disc_linear_policy/instance0.rddl'


# domain_name = 'Navigation_cont'
# domain_path = '../RDDL/Navigation_cont/domain.rddl'
# instance_path = '../RDDL/Navigation_cont/instance0.rddl'

# domain_name_new = 'Navigation_cont_error'
# domain_path_new = '../RDDL/Navigation_cont_error/domain.rddl'
# instance_path_new = '../RDDL/Navigation_cont_error/instance0.rddl'


myEnv = RDDLEnv.RDDLEnv(domain=domain_path, instance=instance_path)

xadd_model = RDDLModelWXADD(myEnv.model)

print(xadd_model.cpfs)

xadd_model.compile()
