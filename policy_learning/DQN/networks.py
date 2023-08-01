import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):

    def __init__(self, state_size, action_size, seed, fc1_units=128, fc2_units=64):

        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)

    #     self.init_weights()

    # def init_weights(self, n=0):
    #     nn.init.constant_(self.fc1.weight, 0) # [-1, 0, 1]
    #     nn.init.constant_(self.fc2.weight, 0) # [-1, 0, 1]
    #     nn.init.constant_(self.fc3.weight, 0) # [-1, 0, 1]

    """
    ###################################################
    Build a network that maps state -> action values.
    """
    def forward(self, state):
        
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)