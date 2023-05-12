# 作者 Ajex
# 创建时间 2023/5/6 0:47
# 文件名 Agent.py
import time

import torch
import torch.nn as nn
import torch.nn.functional as func
import numpy as np

import Environment

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:",device)

class Agent:
    def __init__(self, veh_index, des_index, env: Environment.Environment, memory_size):
        self.veh_index = veh_index
        self.des_index = des_index
        self.env = env
        self.state_dim = 1 + 1 + env.n_sub_carrier + 4 * env.n_sub_carrier + 1 + 1
        # 剩余载荷1+剩余时间1+干扰功率M+信道增益4*M+迭代次数1+探索率1
        self.n_action = env.n_sub_carrier * 4
        # M个信道*4种功率
        self.dqn = DQN(self.state_dim, self.n_action, memory_size, batch_size=64)
        self.transition_buffer = []

    def act(self, local_obs, epsilon, episode):
        # 1. 将状态转化为tensor，并添加“指纹”
        # 2. 将状态输入神经网络，神经网络按照epsilon-greedy给出动作
        # 3. 返回动作
        tensors = [torch.FloatTensor(s).flatten() for s in local_obs]
        tensors.append(torch.FloatTensor([epsilon, episode]))
        state_vector = torch.cat(tensors, dim=0)
        action_num = self.dqn.choose_action(state_vector, epsilon)
        action = (action_num // 4, action_num % 4)
        self.transition_buffer.append(state_vector)
        self.transition_buffer.append(torch.FloatTensor([action_num]))
        return action

    def remember_latest(self, global_reward, new_local_obs):
        tensors = [torch.FloatTensor(s).flatten() for s in new_local_obs]
        # 很烂的代码，但我不想再去改Algorithm里的代码了
        episode, epsilon = self.transition_buffer[0][-2:]
        tensors.append(torch.FloatTensor([epsilon, episode]))
        state_vector = torch.cat(tensors, dim=0)
        self.transition_buffer.append(torch.FloatTensor([global_reward]))
        self.transition_buffer.append(state_vector)
        transition_vector = torch.cat(self.transition_buffer, dim=0)
        self.dqn.add_transition(transition_vector)
        self.transition_buffer.clear()

    def train(self):
        self.dqn.learn_with_min_batch()

    def get_net_state(self):
        return self.dqn.eval_net.state_dict()

    def read_net_state(self, net_state_dict):
        self.dqn.eval_net.load_state_dict(net_state_dict)




class Net(nn.Module):
    def __init__(self, state_dim, n_actions):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(state_dim, 500)
        self.fc1.weight.data.normal_(0, 0.1)
        self.fc2 = nn.Linear(500, 250)
        self.fc2.weight.data.normal_(0, 0.1)
        self.fc3 = nn.Linear(250, 120)
        self.out = nn.Linear(120, n_actions)

    def forward(self, state):
        # Q-Network的输出应为给定状态下的每个状态的q值
        x1 = func.relu(self.fc1(state))
        x2 = func.relu(self.fc2(x1))
        x3 = func.relu(self.fc3(x2))
        q_value = self.out(x3)
        return q_value


class DQN(object):
    def __init__(self, state_dim, n_action, memory_size, batch_size):
        self.eval_net, self.target_net = Net(state_dim, n_action).to(device), Net(state_dim, n_action).to(device)
        self.state_dim, self.action_dim = state_dim, n_action
        self.learn_cnt = 0
        self.TARGET_REPLACE_ITER = 100
        self.memory = ReplayMemory(memory_size, self.state_dim * 2 + 2)
        self.learning_rate = 0.001
        self.optimizer = torch.optim.RMSprop(self.eval_net.parameters(), lr=self.learning_rate)
        self.gamma = 0.9  # 需要进一步确定折扣率
        self.loss_func = nn.MSELoss()
        self.batch_size = batch_size

    def choose_action(self, state_vector, epsilon):
        # epsilon-greedy
        state_vector = state_vector.to(device)
        if np.random.uniform(0, 1) >= epsilon:
            q_value = self.eval_net.forward(state_vector)
            state_idx = torch.argmax(q_value).item()
            return state_idx
        else:
            return np.random.randint(0, self.action_dim)

    def add_transition(self, transition):
        self.memory.add(transition)

    def learn_with_min_batch(self):
        if self.learn_cnt % self.TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_cnt += 1
        transitions = self.memory.choose(self.batch_size)
        # memory中数据按行存储，所以取出来的经验是行优先的，所以需要按列取数据
        old_states = transitions[:, :self.state_dim]
        actions = transitions[:, self.state_dim:self.state_dim + 1].long()
        rewards = transitions[:, self.state_dim + 1:self.state_dim + 2]
        new_states = transitions[:, self.state_dim + 2:]
        # print(old_states.shape)
        # print(actions)
        # print(rewards.shape)
        # print(new_states.shape)
        # print(self.eval_net(old_states).shape)

        q_eval = self.eval_net(old_states).gather(1, actions)
        q_next = self.target_net(new_states).detach()
        max_q = q_next.max(1)[0].view(self.batch_size, 1)
        q_target = rewards + self.gamma * max_q
        loss = self.loss_func(q_eval, q_target)
        # 计算new_states的q值
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


class ReplayMemory:
    def __init__(self, size, transition_dim):
        self.size, self.transition_dim = size, transition_dim
        self.counter = 0
        self.mem = torch.zeros(size, transition_dim).to(device)

    def add(self, transition):
        self.mem[self.counter % self.size] = transition
        self.counter = self.counter + 1

    def choose(self, batch_size):
        sample_index = np.random.choice(min(self.counter, self.size), batch_size)
        return self.mem[sample_index]
