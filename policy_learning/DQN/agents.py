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

    def __init__(self, env, model, context, value_xadd_path='', q_xadd_path='', value_cache_path='', q_cache_path='', dqn_type='DQN', replay_memory_size=1e5, batch_size=64, gamma=0.99,
    	learning_rate=1e-3, tau=0.05, update_rate=4, seed=0, device='cuda:0',agent_type="vanilla", domain_type="None", use_cache=False):
        
        self.env = env
        self.value_xadd_path = value_xadd_path
        self.q_xadd_path = q_xadd_path
        
        self.value_cache_path = value_cache_path
        self.q_cache_path = q_cache_path

        self.model = model
        self.context = context

        if agent_type.lower() != 'dqn':
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
        self.tau = tau
        self.update_rate = update_rate
        self.seed = random.seed(seed)
        self.device = device
        self.agent_type = agent_type
        self.domain_type = domain_type

        self.network = QNetwork(self.state_size, self.action_size, seed).to(self.device)
        self.target_network = QNetwork(self.state_size, self.action_size, seed).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=self.learn_rate)
        self.memory = ReplayBuffer(self.action_size, self.buffer_size, self.batch_size, seed, device)

        self.t_step = 0

        self.use_cache = use_cache
    
        self.value_cache = {}
        self.q_cache = {}

        if self.use_cache:
            self.load_cache()

    def reset(self, seed=0):
        random.seed(seed)
        self.network = QNetwork(self.state_size, self.action_size, seed).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=self.learn_rate)
        self.memory = ReplayBuffer(self.action_size, self.buffer_size, self.batch_size, seed, self.device)
        self.env.reset()

    def load_cache(self):
        for k, v in self.value_cache_path.items():
            with open(v) as data_file:
                value_cache_dict = json.load(data_file)
                self.value_cache[k] = value_cache_dict
        for k, v in self.q_cache_path.items():
            with open(v) as data_file:
                q_xadd_dict = json.load(data_file)
                self.q_cache[k] = q_xadd_dict
        
                    


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
    
    # this function generate a dict which mapes a tuple state values into a value calcualted by XADD
    def gen_value_cache(self):
        state_size = len(self.state_index_dict)
        if "reservoir" in self.domain_type:
            state_range = 100
        elif "navigation" in self.domain_type:
            state_range = 10
        else:
            raise ValueError('{} not implemnted in cache generation'.format(self.domain_type))
        range_list = []
        for r in range(state_size):
            range_list.append(range(0,state_range+1,1))
        
        all_int_states = list(itertools.product(*range_list))

        value_xadd_cache = {}
        q_xadd_cache = {}
        for v_type, v_node in self.value_xadd_nodes.items():
            value_xadd_cache[v_type] = {}
        
        for a_type, q_lst in self.q_xadd_nodes.items():
            temp_q_lst = []
            for t in q_lst:
                temp_q_lst.append((t[0],{}))
            q_xadd_cache[a_type] = temp_q_lst       
        
        for state_tuple in all_int_states:
            state = {v:state_tuple[k] for k,v in self.state_index_dict.items()}
            state_c_assign = {self.context._str_var_to_var[k]:v for k,v in state.items()}
            for v_type, v_node in self.value_xadd_nodes.items():
                state_value = self.context.evaluate(v_node, bool_assign={}, cont_assign=state_c_assign)
                value_xadd_cache[v_type][str(state_tuple)] = str(state_value)
            for q_type, q_lst in self.q_xadd_nodes.items():
                for i, q in enumerate(q_lst):
                    q_value = self.context.evaluate(q[1], bool_assign={}, cont_assign=state_c_assign)
                    q_xadd_cache[q_type][i][1][str(state_tuple)] = str(q_value)

        return value_xadd_cache, q_xadd_cache


    def e_greedy_action(self, action_values, eps):
        if random.random() > eps:
            action_tensor = np.argmax(action_values.cpu().data.numpy())
        else:
            action_tensor = random.choice(np.arange(self.action_size))
        return action_tensor
    
    # ppr select source policy using q value of previous source MDP using psi, psi is discounted every step
    def ppr_action(self, state, action_values, eps, psi):
        best_action_val = -np.inf
        if np.random.random() < psi:
            for i, a in self.action_index_dict.items():
                if self.use_cache:
                    state_str = str(tuple([int(i) for i in self.state_to_vec(state)]))
                    q_v = float([i[1][state_str] for i in self.q_cache['q_source'] if i[0] == a][0])
                else:
                    state_c_assign = {self.context._str_var_to_var[k]:v for k,v in state.items()}
                    q_source_node = [i[1] for i in self.q_xadd_nodes['q_source'] if i[0] == a][0]
                    q_v = self.context.evaluate(q_source_node, bool_assign={}, cont_assign=state_c_assign)
                if q_v >= best_action_val:
                    best_action_val = q_v
                    best_action_idx = i
            return best_action_idx
            
        else:
            if random.random() > eps:
                action = np.argmax(action_values.cpu().data.numpy())
            else:
                action = random.choice(np.arange(self.action_size))
            return action
        
    def baseline_action(self, state):
        best_action_val = -np.inf
        if 'source' in self.agent_type:
            policy = 'q_source'
        elif 'target' in self.agent_type:
            policy = 'q_target'
        for i, a in self.action_index_dict.items():
            if self.use_cache:
                state_str = str(tuple([int(i) for i in self.state_to_vec(state)]))
                q_v = float([i[1][state_str] for i in self.q_cache[policy] if i[0] == a][0])
            else:
                state_c_assign = {self.context._str_var_to_var[k]:v for k,v in state.items()}
                q_source_node = [i[1] for i in self.q_xadd_nodes[policy] if i[0] == a][0]
                q_v = self.context.evaluate(q_source_node, bool_assign={}, cont_assign=state_c_assign)
            if q_v >= best_action_val:
                best_action_val = q_v
                best_action_idx = i
        return best_action_idx
            



    def step(self, state, action, reward, next_state, done):
        
        self.memory.add(state, action, reward, next_state, done)

        self.t_step = (self.t_step + 1) % self.update_rate
        if self.t_step == 0:
            if len(self.memory) > self.batch_size:
                experiences = self.memory.sample()
                if self.agent_type not in ["source_policy", "target_policy"]:
                    self.learn(experiences, self.gamma)


    def act(self, state, eps, psi):
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
            action_vec = self.ppr_action(state, action_values, eps, psi)
        elif self.agent_type == "source_policy" or self.agent_type == "target_policy":
            action_vec = self.baseline_action(state)
        else:
            action_vec = self.e_greedy_action(action_values, eps)

        action = self.vec_to_action(action_vec)

        return state_vec, action_vec, action

    # shaped reward use value function from the srouce MDP as 
    def cal_potential_diff(self, states, actions, next_states, shape):
        shaped_reward_list = []
        for s, a, ns in zip(states.detach().cpu().numpy(), actions.detach().cpu().numpy(), next_states.detach().cpu().numpy()):
            
            state_vec = list(s)
            state = self.vec_to_state(state_vec)
            state_c_assign = {self.context._str_var_to_var[k]:v for k,v in state.items()}

            next_state_vec = list(ns)
            action_vec = a
            next_state = self.vec_to_state(next_state_vec)
            next_state_c_assign = {self.context._str_var_to_var[k]:v for k,v in next_state.items()}

            if self.use_cache:
                state_str = str(tuple([int(i) for i in state_vec]))
                next_state_str = str(tuple([int(i) for i in next_state_vec]))
                potential_current_s = float(self.value_cache['v_source'][state_str])
                # potential_current_s = float(self.value_cache['v_target'][state_str])

                potential_next_s = float(self.value_cache['v_source'][next_state_str])
                # potential_next_s = float(self.value_cache['v_target'][next_state_str])
            else:
                v_source_node = self.value_xadd_nodes['v_source']
                # v_diff_node = self.value_xadd_nodes['v_diff']
                # v_target_node = self.value_xadd_nodes['v_target']
                v_node = v_source_node

                potential_current_s = self.context.evaluate(v_node, bool_assign={}, cont_assign=state_c_assign)
                potential_next_s = self.context.evaluate(v_node, bool_assign={}, cont_assign=next_state_c_assign)

            shaped_reward = self.gamma * potential_next_s - potential_current_s

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

            if self.use_cache:
                state_str = str(tuple([int(i) for i in state_vec]))
                # q_source_v = float([i[1][state_str] for i in self.q_cache['q_source'] if i[0] == action][0])
                # q_diff_v = float([i[1][state_str] for i in self.q_cache['q_diff'] if i[0] == action][0])
                q_target_v = float([i[1][state_str] for i in self.q_cache['q_target'] if i[0] == action][0])

            else:
                q_source_node = [i[1] for i in self.q_xadd_nodes['q_source'] if i[0] == action][0]
                q_diff_node = [i[1] for i in self.q_xadd_nodes['q_diff'] if i[0] == action][0]
                # q_target_node = [i[1] for i in self.q_xadd_nodes['q_target'] if i[0] == action][0]

                q_source_v = self.context.evaluate(q_source_node, bool_assign={}, cont_assign=state_c_assign)
                q_diff_v = self.context.evaluate(q_diff_node, bool_assign={}, cont_assign=state_c_assign)
                # q_target_v = self.context.evaluate(q_target_node, bool_assign={}, cont_assign=state_c_assign)

            # lowerbound = q_source_v + q_diff_v
            lowerbound = q_target_v
            # lowerbound = q_source_v

            lowerbound_list.append(lowerbound)

        ##########################################################################################

        # ## use v diff to calculate q legency code for backup
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



    def gen_counterfactual_values(self, states, all_Qs, actions):


        target_Qs = np.zeros(all_Qs.shape)        

        # p_set = set()

        # use using q value
        s_idx = 0
        for s, a in zip(states.detach().cpu().numpy(), actions.detach().cpu().numpy()):
            state_vec = list(s)
            action_vec = a[0]
            state = self.vec_to_state(state_vec)
            state_c_assign = {self.context._str_var_to_var[k]:v for k,v in state.items()}
            action = self.vec_to_action(action_vec)

            for a in self.action_index_dict.keys():
                if a == action_vec:
                    target_Qs[s_idx][a] == all_Qs[s_idx][a]
                else:
                    if self.use_cache:
                        state_str = str(tuple([int(i) for i in state_vec]))
                        # q_source_v = float([i[1][state_str] for i in self.q_cache['q_source'] if i[0] == action][0])
                        # q_diff_v = float([i[1][state_str] for i in self.q_cache['q_diff'] if i[0] == action][0])
                        q_target_v = float([i[1][state_str] for i in self.q_cache['q_target'] if i[0] == action][0])

                    else:
                        q_source_node = [i[1] for i in self.q_xadd_nodes['q_source'] if i[0] == action][0]
                        q_diff_node = [i[1] for i in self.q_xadd_nodes['q_diff'] if i[0] == action][0]
                        # q_target_node = [i[1] for i in self.q_xadd_nodes['q_target'] if i[0] == action][0]

                        q_source_v = self.context.evaluate(q_source_node, bool_assign={}, cont_assign=state_c_assign)
                        q_diff_v = self.context.evaluate(q_diff_node, bool_assign={}, cont_assign=state_c_assign)
                        # q_target_v = self.context.evaluate(q_target_node, bool_assign={}, cont_assign=state_c_assign)
                    lowerbound = q_target_v     
                    target_Qs[s_idx][a] = max([lowerbound,  all_Qs[s_idx][a]])
            s_idx += 1
            
        return target_Qs



    
    def learn(self, experiences, gamma, DQN=True):
        
        states, actions, rewards, next_states, dones = experiences

                
        if "counterfactual" in self.agent_type:
            all_Qs = self.network(states)
            cf_update = self.gen_counterfactual_values(states, all_Qs, actions)
            cf_update = torch.from_numpy(cf_update).float().to(self.device)
            for i in self.action_index_dict.keys():
                all_Qs = self.network(states)
                idx_tensor = torch.full(actions.size(), i).to(self.device)
                qsa = all_Qs.gather(1, idx_tensor).clone()
                qsa_target = cf_update.gather(1, idx_tensor).clone()
                loss = F.mse_loss(qsa, qsa_target)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()


        all_Qs = self.network(states)
        Qsa = self.network(states).gather(1, actions)

        if (self.dqn_type == 'DQN'):
            Qsa_prime_target_values = self.target_network(next_states).detach()
            Qsa_prime_targets = Qsa_prime_target_values.max(1)[0].unsqueeze(1)
        
        # Compute Q targets for current states 
        Qsa_targets = rewards + (gamma * Qsa_prime_targets * (1 - dones))

        if  'lowerbound' in self.agent_type:
            lowerbounds = self.cal_lowerbound(states, actions, rewards, next_states, shape=Qsa_targets.shape)
            y = torch.maximum(lowerbounds, Qsa_targets)
            Qsa_targets = y
        elif 'rewardshaping' in self.agent_type:
            reward_potential = self.cal_potential_diff(states, actions, next_states, shape=Qsa_targets.shape)
            Qsa_targets = Qsa_targets + reward_potential


        # Compute loss (error)
        loss = F.mse_loss(Qsa, Qsa_targets)

        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.soft_update(self.network, self.target_network, self.tau)

    

    def soft_update(self, local_model, target_model, tau):

        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

    
    


