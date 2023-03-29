from modeldiff.utils.utils import get_xadd_model_from_file
import os

class ModelDiff:
    def __init__(self, 
                model_1_path:str, model_2_path:str,
                inst_1_path:str=None,
                inst_2_path:str=None):
        self._model_1_path = model_1_path
        self._inst_1_path = inst_1_path
        self._model_2_path = model_2_path
        self._inst_2_path = inst_2_path
        self._model_1, self._context_1 = get_xadd_model_from_file(model_1_path, inst_1_path=None)
        self._model_2, self._context_2 = get_xadd_model_from_file(model_2_path, inst_2_path=None)
        self._model_diff = None
        self._context_diff = None
    
    def build_model_with_diff_reward(self):
        self._model_diff, self._context_diff = get_xadd_model_from_file(self._model_2_path, self._inst_2_path)
        r1_path = self._context_1.export_xadd(xadd_model_1.reward, 'temp1.xadd')
        r1_node = self._context_diff.import_xadd(fname='temp1.xadd', locals=context1._str_var_to_var)
        os.remove(r1_path)
        diff_node = self._context_diff.apply(self.model_diff.reward, r1_node, 'subtract')





m1_path = "../RDDL/Navigation_disc_goal_551010/domain.rddl"
m2_path = "../RDDL/Navigation_disc_goal_771010/domain.rddl"

model_diff = ModelDiff(m1_path, m2_path)

    

