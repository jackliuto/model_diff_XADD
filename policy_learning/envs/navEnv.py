import gym
from gym.spaces import Discrete, Dict, Box
import numpy as np




class envWrapperNav(gym.Env):
    def __init__(self, RDDLEnv):
        super(envWrapperNav, self).__init__()
        self.RDDLEnv = RDDLEnv
        self.state = RDDLEnv.reset()
        self.action_list = sorted(self.RDDLEnv.action_space.keys())
        self.observation_space = self.RDDLEnv.observation_space
        self.action_space = Discrete(len(self.action_list))
        self.max_timesteps = 1000
    
    def action_vec2dict(self, action):
        action_dict = {}
        for idx, action in enumerate(self.action_list):
            v = 1 if idx == action else 0
            action_dict[action] = v
        return action_dict

    def reset(self):
        state = self.RDDLEnv.reset()
        self.current_step = 0
        return state
    
    def step(self, action):
        action_dict = self.action_vec2dict(action)
        next_state, reward, _, info = self.RDDLEnv.step(action_dict)
        done = self.current_step >= self.max_episode_length
        self.state = next_state   
        return self.state, reward, done, info  # Modify as needed

