# 作者 Ajex
# 创建时间 2023/5/8 15:54
# 文件名 Algorithm.py
import os

import numpy as np
import Environment
import Agent
import torch
import logging


class Algorithm:
    def __init__(self):
        self.B = 2 * 1060  # B
        self.env = Environment.Environment(4, 1, self.B)
        self.agents = []
        self.T = 0.1
        self.agent_time_step = 0.001
        self.D = 1000000
        self.lambda_V2I = 1 / (1 + self.env.n_veh * self.env.n_des)
        self.lambda_V2V = 1 - self.lambda_V2I
        self.n_episodes = 3000
        self.beta = 12.5  # MBps
        self.init_logger()

    def init_logger(self):
        self.logger = logging.getLogger('Algorithm')
        fh = logging.FileHandler('2023-5-11_4_1.log')
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)
        self.logger.setLevel(logging.INFO)

    def train(self):
        print('训练开始')

        def random_actions():
            channels = np.random.randint(0, self.env.n_sub_carrier, size=(self.env.n_veh, self.env.n_des))
            power_selections = np.random.randint(0, len(self.env.V2V_power_dB_list),
                                                 size=(self.env.n_veh, self.env.n_des))
            return np.stack((channels, power_selections), axis=2)

        for i in range(self.env.n_veh):
            for j in range(self.env.n_des):
                self.agents.append(Agent.Agent(i, j, self.env, self.D))

        steps = int(self.T / self.agent_time_step)
        for e in range(self.n_episodes):
            # 计算epsilon
            epsilon = get_epsilon(e)
            self.env.renew_environment()
            self.env.update_small_fading()
            self.env.reset_payload(self.B)
            self.env.reset_time(self.T)
            V2I_sum_capacity = 0
            reward_sum = 0
            all_actions = random_actions()  # 随机初始化所有actions，即随机选择信道和功率
            for t in range(steps):
                global_obs = self.env.get_observations(all_actions)
                for agent in self.agents:
                    local_obs = self.env.get_local_observation(global_obs, agent.veh_index, agent.des_index)
                    action = agent.act(local_obs, epsilon, e)
                    all_actions[agent.veh_index][agent.des_index] = action
                V2I_capacity = self.env.get_V2I_capacity(all_actions)
                V2I_sum_capacity += sum(V2I_capacity) / (8 * 1000 * 1000)
                V2V_capacity = self.env.get_V2V_capacity(all_actions)
                global_reward = self.compute_reward(V2I_capacity, V2V_capacity)
                reward_sum += global_reward
                self.env.update_small_fading()
                new_global_obs = self.env.get_observations(all_actions)
                for agent in self.agents:
                    # 建立回放区，存储经验
                    new_local_obs = self.env.get_local_observation(new_global_obs, agent.veh_index, agent.des_index)
                    agent.remember_latest(global_reward, new_local_obs)
            # print("本轮剩余载荷：\n", self.env.remain_payload)
            info = "episode - %d" % e + "  " + "V2I平均速率：%fMBps" % (V2I_sum_capacity / steps) + "  " \
                   + "V2V交付成功占比：%f" % \
                   (sum([1 if payload <= 0 else 0 for payload in self.env.remain_payload.reshape(
                       (self.env.n_veh * self.env.n_des))]) / (self.env.n_veh * self.env.n_des)
                    ) + "  " \
                   + "平均奖励值：%fMBps" % (reward_sum / steps)

            self.logger.info(info)
            for agent in self.agents:
                agent.train()

    def compute_reward(self, V2I_capacity, V2V_capacity):
        # 为避免奖励值过大，单位换为MBps
        reward = np.sum(V2I_capacity) / (8 * 1000 * 1000) * self.lambda_V2I
        reward += sum([V2V_capacity[i][j] / (8 * 1000 * 1000) if self.env.remain_payload[i][j] > 0 else self.beta
                       for i in range(self.env.n_veh)
                       for j in range(self.env.n_des)]) * self.lambda_V2V
        return reward

    def save_model(self, model_root_dir):
        if model_root_dir[-1] != '/':
            model_root_dir += '/'
        model_dir = model_root_dir + 'env%d_%d/' % (self.env.n_veh, self.env.n_des)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        for agent in self.agents:
            model = agent.get_net_state()
            path = model_dir + "agent%d_%d.m" % (agent.veh_index, agent.des_index)
            torch.save(model, path)

    def test(self, model_root_dir, rounds, payload_size):
        def random_actions():
            channels = np.random.randint(0, self.env.n_sub_carrier, size=(self.env.n_veh, self.env.n_des))
            power_selections = np.random.randint(0, len(self.env.V2V_power_dB_list),
                                                 size=(self.env.n_veh, self.env.n_des))
            return np.stack((channels, power_selections), axis=2)

        model_dir = model_root_dir + 'env%d_%d/' % (self.env.n_veh, self.env.n_des)
        if not os.path.exists(model_dir):
            print("目录不存在")
            return
        for i in range(self.env.n_veh):
            for j in range(self.env.n_des):
                path = model_dir + "agent%d_%d.m" % (i, j)
                if not os.path.exists(path):
                    print("模型文件不存在")
                    return
                agent = Agent.Agent(i, j, self.env, self.D)
                self.agents.append(agent)
                agent.read_net_state(torch.load(path))
        steps = int(self.T / self.agent_time_step)
        for e in rounds:
            self.env.renew_environment()
            self.env.update_small_fading()
            self.env.reset_payload(payload_size)
            self.env.reset_time(self.T)
            all_actions = random_actions()
            for t in range(steps):
                global_obs = self.env.get_observations(all_actions)
                for agent in self.agents:
                    local_obs = self.env.get_local_observation(global_obs, agent.veh_index, agent.des_index)
                    action = agent.act(local_obs, 1, e)
                    all_actions[agent.veh_index][agent.des_index] = action
                V2I_capacity = self.env.get_V2I_capacity(all_actions)
                V2V_capacity = self.env.get_V2V_capacity(all_actions, True)
            # 需要统计哪些数据？


def get_epsilon(x):
    if x < 2400:
        return 1 - 0.98 / 2399 * x
    else:
        return 0.02


if __name__ == '__main__':
    algorithm = Algorithm()
    algorithm.n_episodes = 3000
    algorithm.train()
    algorithm.save_model('./models/')
