# 作者 Ajex
# 创建时间 2023/5/6 0:47
# 文件名 Agent.py
import torch
import torch.nn as nn
import torch.nn.functional as func
import numpy as np

import Environment
import Memory


class Agent:
    def __init__(self, veh_index, des_index, env: Environment.Environment, memory_size):
        self.veh_index = veh_index
        self.des_index = des_index
        self.env = env
        self.state_dim = 1 + 1 + env.n_sub_carrier + 4 * env.n_sub_carrier
        self.action_dim = 4 * env.n_sub_carrier
        self.dqn = DQN(self.state_dim, self.action_dim, memory_size)

    def act(self, local_state):
        # 1. 将状态转化为tensor
        # 2. 将状态输入神经网络，神经网络按照epsilon-greedy给出动作
        # 3. 返回动作
        tensors = [torch.tensor(s).flatten() for s in local_state]
        state_vector = torch.cat(tensors, dim=0)
        action_num = self.dqn.choose_action(state_vector)
        action = (action_num // self.env.n_sub_carrier, action_num % self.env.n_sub_carrier)
        return action

    def add_transition(self, local_state, action, global_reward, new_local_state):
        self.dqn.add_transition()
        pass


class Net(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(state_dim, 500)
        self.fc1.weight.data.normal_(0, 0.1)
        self.fc2 = nn.Linear(500, 250)
        self.fc2.weight.data.normal_(0, 0.1)
        self.fc3 = nn.Linear(250, 120)
        self.out = nn.Linear(120, action_dim)

    def forward(self, state):
        # Q-Network的输出应为给定状态下的每个状态的q值
        x1 = func.relu(self.fc1(state))
        x2 = func.relu(self.fc2(x1))
        x3 = func.relu(self.fc3(x2))
        q_value = self.out(x3)
        return q_value


class DQN(object):
    def __init__(self, state_dim, action_dim, memory_size):
        self.eval_net, self.target_net = Net(state_dim, action_dim), Net(state_dim, action_dim)
        self.state_dim, self.action_dim = state_dim, action_dim
        self.step_cnt = 0
        self.memory = Memory.ReplayMemory(memory_size)
        self.learning_rate = 0.001
        self.epsilon = 1
        self.loss_func = nn.MSELoss()

    def choose_action(self, state_vector):
        # epsilon-greedy
        if np.random.uniform(0, 1) < self.epsilon:
            q_value = self.eval_net.forward(state_vector)
            state_idx = torch.argmax(q_value).item()
            return state_idx
        else:
            return np.random.randint(0, self.action_dim)
