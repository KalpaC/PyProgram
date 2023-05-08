# 作者 Ajex
# 创建时间 2023/5/8 15:54
# 文件名 Algorithm.py
import numpy as np

import Environment
import Agent


class Algorithm:
    def __init__(self, B):
        self.env = Environment.Environment()
        self.env.standard_init(4, 4, 1060)
        self.agents = []
        self.B = B
        self.T = 0.1
        self.agent_time_step = 0.001
        for i in range(self.env.n_veh):
            for j in range(self.env.n_des):
                self.agents.append(Agent.Agent(i, j))

    def train(self, n_episodes):
        def random_actions():
            channels = np.random.randint(0, self.env.n_sub_carrier, size=(self.env.n_veh, self.env.n_des))
            power_selections = np.random.randint(0, len(self.env.V2V_power_dB_list),
                                                 size=(self.env.n_veh, self.env.n_des))
            return np.stack((channels,power_selections),axis=2)

        for i in range(n_episodes):
            self.env.renew_environment()
            self.env.reset_payload(self.B)
            self.env.reset_time(self.T)
            all_actions = random_actions()  # 随机初始化所有actions
            pass
            for t in range(int(self.T / self.agent_time_step)):
                states = self.env.get_states(all_actions)
                for agent in self.agents:
                    local_state = self.env.get_local_state(states, agent.veh_index, agent.des_index)
                    action = agent.act(local_state)
                    all_actions[agent.veh_index][agent.des_index] = action
                global_reward = self.env.get_reward(all_actions, 0.5, 0.5)
                self.env.update_small_fading()
                new_states = self.env.get_states(all_actions)
                for agent in self.agents:
                    # 建立回放区，存储经验
                    new_local_state = self.env.get_local_state(new_states, agent.veh_index, agent.des_index)
                    action = all_actions[agent.veh_index][agent.des_index]
                    agent.add_transition(local_state, action, global_reward, new_local_state)
            for agent in self.agents:
                agent.train_with_mini_batch()

    def save_model(self):
        pass

    def predict(self):
        pass
