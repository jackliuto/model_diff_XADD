import torch
import torch.nn.functional as F
import torch.optim as optim

import itertools
import json

import numpy as np
import random
from collections import namedtuple, deque

from xaddpy import XADD


from .networks import QNetwork
from .memory import ReplayBuffer

class DQN_Agent():

    def __init__(self, env, model, context, value_xadd_path, q_xadd_path, dqn_type='DQN', replay_memory_size=1e5, batch_size=64, gamma=0.99,
    	learning_rate=1e-3, target_tau=2e-3, update_rate=4, seed=0, device='cuda:0',agent_type="vanilla"):
        
        self.env = env
        self.value_xadd_path = value_xadd_path
        self.q_xadd_path = q_xadd_path
        self.model = model
        self.context = context

        self.value_xadd_nodes = self.create_value_xadd(self.context, self.value_xadd_path)
        self.q_xadd_nodes = self.create_q_xadd(self.context, self.q_xadd_path)

        # print(self.value_xadd_nodes)
        # print(self.q_xadd_nodes)

        self.action_name_list = [i for i in self.env.action_space.keys()]
        self.action_name_list.sort()
        self.state_name_list = [i for i in self.env.observation_space.keys()]
        self.state_name_list.sort()
        self.state_size = len(self.state_name_list)
        self.action_size = 2**(len(self.action_name_list))
        self.state_index_dict = self.gen_state_index_dict(self.state_name_list)
        self.action_index_dict = self.gen_action_index_dict(self.action_name_list)
        self.dqn_type = 'DQN'
        self.buffer_size = int(replay_memory_size)
        self.batch_size = batch_size
        self.gamma = gamma
        self.learn_rate = learning_rate
        self.tau = target_tau
        self.update_rate = update_rate
        self.seed = random.seed(seed)
        self.device = device
        self.agent_type = agent_type

        self.network = QNetwork(self.state_size, self.action_size, seed).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=self.learn_rate)
        self.memory = ReplayBuffer(self.action_size, self.buffer_size, self.batch_size, seed, device)

        self.t_step = 0

        self.psi = 0.5
    
    def create_value_xadd(self, context, value_xadd_path):
        value_xadd_nodes = {}        
        for k, v in value_xadd_path.items():
            with open(v) as data_file:
                value_xadd_dict = json.load(data_file)
            for k, v in value_xadd_dict.items():
                node_id = self.context.import_xadd(xadd_str=v)
                value_xadd_nodes[k] = node_id
        return value_xadd_nodes
    
    def create_q_xadd(self, context, q_xadd_path):
        q_xadd_nodes = {}        
        for k, v in q_xadd_path.items():
            with open(v) as data_file:
                q_xadd_list = json.load(data_file)
            q_list = []
            for i in q_xadd_list:
                action = i[0]
                node_id = self.context.import_xadd(xadd_str=i[1])
                q_list.append((action, node_id))
            q_xadd_nodes[k] = q_list
        return q_xadd_nodes

    def gen_state_index_dict(self, state_name_list):
        state_index_dict = {}
        for i, v in enumerate(state_name_list):
            state_index_dict[i] = v
        return state_index_dict
    
    def gen_action_index_dict(self, action_name_list):
        action_index_dict = {}
        bool_combos = [list(i) for i in itertools.product([0, 1], repeat=len(action_name_list))]
        action_list = []
        for b in bool_combos:
            a = {}
            for i, v in enumerate(b):
                a[action_name_list[i]] = True if v==1 else False
            action_list.append(a)
        for i, v in enumerate(action_list):
            action_index_dict[i] = v
        return action_index_dict

    def state_to_vec(self, state):
        state_tensor = np.zeros(self.state_size)
        for k, v in self.state_index_dict.items():
            index = k
            value = state[v]
            state_tensor[index] = value

        return state_tensor

    def vec_to_action(self, action_vec):
        for k, v in self.action_index_dict.items():
            if k == action_vec:
                return v
    
    def vec_to_state(self, state_vec):
        return {self.state_index_dict[i]:v for i,v in enumerate(state_vec)}


    def e_greedy_action(self, action_values, eps):
        if random.random() > eps:
            action_tensor = np.argmax(action_values.cpu().data.numpy())
        else:
            action_tensor = random.choice(np.arange(self.action_size))
        return action_tensor
    
    def ppr_action(self, state, action_values, eps, psi):
        best_action_val = -99999999999
        if np.random.random() < psi:
            self.psi = psi*0.9
            state_c_assign = {self.context._str_var_to_var[k]:v for k,v in state.items()}
            for i, a in self.action_index_dict.items():
                q_source_node = [i[1] for i in self.q_xadd_nodes['q_source'] if i[0] == a][0]
                q_v = self.context.evaluate(q_source_node, bool_assign={}, cont_assign=state_c_assign)
                if q_v >= best_action_val:
                    best_action_val = q_v
                    best_action_idx = i
            return best_action_idx
            
        else:
            if random.random() > eps:
                action_tensor = np.argmax(action_values.cpu().data.numpy())
            else:
                action_tensor = random.choice(np.arange(self.action_size))
            return action_tensor
    





    def step(self, state, action, reward, next_state, done):
        
        self.memory.add(state, action, reward, next_state, done)

        self.t_step = (self.t_step + 1) % self.update_rate
        if self.t_step == 0:
            if len(self.memory) > self.batch_size:
                experiences = self.memory.sample()
                self.learn(experiences, self.gamma)


    def act(self, state, eps=0.0):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state_vec = self.state_to_vec(state)
        state_tensor = torch.from_numpy(state_vec).float().unsqueeze(0).to(self.device)
        self.network.eval()
        with torch.no_grad():
            action_values = self.network(state_tensor)
        self.network.train()
        if self.agent_type == "ppr":
            action_vec = self.ppr_action(state, action_values, eps, self.psi)
        else:
            action_vec = self.e_greedy_action(action_values, eps)

        action = self.vec_to_action(action_vec)

        return state_vec, action_vec, action

    def cal_shaped_reward(self, states, actions, next_states, shape):
        shaped_reward_list = []
        for s, a, ns in zip(states.detach().cpu().numpy(), actions.detach().cpu().numpy(), next_states.detach().cpu().numpy()):
            
            state_vec = list(s)
            state = self.vec_to_state(state_vec)
            state_c_assign = {self.context._str_var_to_var[k]:v for k,v in state.items()}

            next_state_vec = list(ns)
            action_vec = a
            next_state = self.vec_to_state(next_state_vec)
            next_state_c_assign = {self.context._str_var_to_var[k]:v for k,v in next_state.items()}

            v_source_node = self.value_xadd_nodes['v_source']
            # v_diff_node = self.value_xadd_nodes['v_diff']
            # v_target_node = self.value_xadd_nodes['v_target']
            v_node = v_source_node

            potential_current_s = self.context.evaluate(v_node, bool_assign={}, cont_assign=state_c_assign)
            potential_next_s = self.context.evaluate(v_node, bool_assign={}, cont_assign=next_state_c_assign)

            shaped_reward = potential_next_s - potential_current_s

            shaped_reward_list.append(shaped_reward)
        
        rs_tensor = np.array(shaped_reward_list, dtype='float32').reshape(shape)
        rs_tensor = torch.as_tensor(rs_tensor).to(self.device)

        return rs_tensor

        
    
    def cal_lowerbound(self, states, actions, rewards, next_states, shape):
        lowerbound_list = []

        # p_set = set()

        # use using q value
        for s, a in zip(states.detach().cpu().numpy(), actions.detach().cpu().numpy()):
            state_vec = list(s)
            action_vec = a
            state = self.vec_to_state(state_vec)
            state_c_assign = {self.context._str_var_to_var[k]:v for k,v in state.items()}

            action = self.vec_to_action(action_vec)

            q_source_node = [i[1] for i in self.q_xadd_nodes['q_source'] if i[0] == action][0]
            q_diff_node = [i[1] for i in self.q_xadd_nodes['q_diff'] if i[0] == action][0]
            # q_target_node = [i[1] for i in self.q_xadd_nodes['q_target'] if i[0] == action][0]
            
            q_source_v = self.context.evaluate(q_source_node, bool_assign={}, cont_assign=state_c_assign) 
            q_diff_v = self.context.evaluate(q_diff_node, bool_assign={}, cont_assign=state_c_assign)
            # q_target_v = self.context.evaluate(q_target_node, bool_assign={}, cont_assign=state_c_assign)

            lowerbound = q_source_v + q_diff_v
            # lowerbound = q_target_v
            # lowerbound = q_source_v

            lowerbound_list.append(lowerbound)

        ##########################################################################################

        # ## use v diff to calculate q
        # for s, a, r, ns in zip(states.detach().cpu().numpy(), actions.detach().cpu().numpy(), rewards, next_states.detach().cpu().numpy()):

        #     next_state_vec = list(ns)
        #     action_vec = a
        #     next_state = self.vec_to_state(next_state_vec)
        #     next_state_c_assign = {self.context._str_var_to_var[k]:v for k,v in next_state.items()}

        #     v_source_node = self.value_xadd_nodes['v_source']
        #     v_diff_node = self.value_xadd_nodes['v_diff']

        #     lowerbound = r + self.gamma*(self.context.evaluate(v_source_node, bool_assign={}, cont_assign=next_state_c_assign) + \
        #                      self.context.evaluate(v_diff_node, bool_assign={}, cont_assign=next_state_c_assign))             


        #     # q_source_node = [i[1] for i in self.q_xadd_nodes['q_source'] if i[0] == action][0]
        #     # q_diff_node = [i[1] for i in self.q_xadd_nodes['q_diff'] if i[0] == action][0]
            
        #     # lowerbound = self.context.evaluate(q_source_node, bool_assign={}, cont_assign=state_c_assign) + \
        #     #              self.context.evaluate(q_diff_node, bool_assign={}, cont_assign=state_c_assign)

        #     lowerbound_list.append(lowerbound)

        
        lb_tensor = np.array(lowerbound_list, dtype='float32').reshape(shape)
        lb_tensor = torch.as_tensor(lb_tensor).to(self.device)


        

        return lb_tensor




        

    
    def learn(self, experiences, gamma, DQN=True):
        
        states, actions, rewards, next_states, dones = experiences

        Qsa = self.network(states).gather(1, actions)

        if (self.dqn_type == 'DQN'):
            Qsa_prime_target_values = self.network(next_states).detach()
            Qsa_prime_targets = Qsa_prime_target_values.max(1)[0].unsqueeze(1)
        
        # Compute Q targets for current states 
        Qsa_targets = rewards + (gamma * Qsa_prime_targets * (1 - dones))

        if self.agent_type =='lowerbound':
            lowerbounds = self.cal_lowerbound(states, actions, rewards, next_states, shape=Qsa_targets.shape)
            y = torch.maximum(lowerbounds, Qsa_targets)
            Qsa_targets = y
        elif self.agent_type =='rewardshaping':
            shaped_reward = self.cal_shaped_reward(states, actions, next_states, shape=Qsa_targets.shape)
            Qsa_targets = Qsa_targets + shaped_reward
        
        # Compute loss (error)
        loss = F.mse_loss(Qsa, Qsa_targets)

        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    
    


