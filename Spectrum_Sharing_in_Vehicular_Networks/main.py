# process2 2023/5/15 14:04

# main 2023/5/15 13:27

import os
import time

import numpy as np
import Environment
import Agent
import torch
import logging
import matplotlib.pyplot as plt


class Algorithm:
    def __init__(self, logger_path):
        self.agents = []
        self.env = None
        self.T = 0.1
        self.agent_time_step = 0.001
        self.D = 1000000
        self.n_episodes = 3000
        self.init_logger(logger_path)
        self.loss_list = []
        self.reward_list = []
        self.V2I_rate_list = []
        self.p_list = []

    def init_logger(self, logger_path):
        self.logger = logging.getLogger('Algorithm')
        fh = logging.FileHandler(logger_path)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)
        self.logger.setLevel(logging.INFO)

    def train(self, n_veh, n_des, payload_size, V2I_weight, V2V_weight, success_reward):
        # 需要统计的数据包括：loss、奖励值、V2I信道容量随episode的变化趋势

        self.env = Environment.Environment(n_veh, n_des, payload_size)

        def random_actions():
            channels = np.random.randint(0, self.env.n_sub_carrier, size=(self.env.n_veh, self.env.n_des))
            power_selections = np.random.randint(0, len(self.env.V2V_power_dB_list),
                                                 size=(self.env.n_veh, self.env.n_des))
            return np.stack((channels, power_selections), axis=2)

        for i in range(n_veh):
            for j in range(n_des):
                self.agents.append(Agent.Agent(i, j, n_veh, self.D))
        steps = int(self.T / self.agent_time_step)
        for e in range(self.n_episodes):
            # 计算epsilon
            epsilon = get_epsilon(e)
            self.env.renew_environment()
            if e == 0:
                self.env.update_small_fading()
            else:
                self.env.compute_channels_with_fastfading()
            self.env.reset_payload(payload_size)
            self.env.reset_time(self.T)
            all_actions = random_actions()  # 随机初始化所有actions，即随机选择信道和功率
            V2I_temp_list = []  # 用于求每个episode的均值
            reward_temp_list = []
            for t in range(steps):
                global_obs = self.env.get_observations(all_actions)
                for agent in self.agents:
                    local_obs = self.env.get_local_observation(global_obs, agent.veh_index, agent.des_index)
                    action = agent.act(local_obs, epsilon, e)
                    all_actions[agent.veh_index][agent.des_index] = action
                V2I_capacity = self.env.get_V2I_capacity(all_actions)
                V2V_capacity = self.env.get_V2V_capacity(all_actions)
                global_reward = self.compute_reward(V2I_capacity, V2V_capacity, V2I_weight, V2V_weight, success_reward)
                # 指标信息存入
                V2I_temp_list.append(V2I_capacity.sum())
                reward_temp_list.append(global_reward)

                self.env.update_small_fading()
                new_global_obs = self.env.get_observations(all_actions)
                for agent in self.agents:
                    # 建立回放区，存储经验
                    new_local_obs = self.env.get_local_observation(new_global_obs, agent.veh_index, agent.des_index)
                    agent.remember_latest(global_reward, new_local_obs)

            for i in range(len(self.agents)):
                loss = self.agents[i].train()
                if i == 0:
                    self.loss_list.append(loss)
            if e % 100 == 0:
                print('episode: %d, agent0 loss: %d' % (e, self.loss_list[-1]))
            self.p_list.append(np.count_nonzero(self.env.remain_payload <= 0) / (n_veh * n_des))
            self.V2I_rate_list.append(sum(V2I_temp_list) / steps)
            self.reward_list.append(sum(reward_temp_list) / steps)

    def compute_reward(self, V2I_capacity, V2V_capacity, V2I_weight, V2V_weight, success_reward):
        # 为避免奖励值过大，单位换为MBps
        reward = np.sum(V2I_capacity) / (8 * 1000 * 1000) * V2I_weight
        reward += sum([V2V_capacity[i][j] / (8 * 1000 * 1000) if self.env.remain_payload[i][j] > 0 else success_reward
                       for i in range(self.env.n_veh)
                       for j in range(self.env.n_des)]) * V2V_weight
        return reward

    def test(self, model_root_dir, cycles, rounds, n_veh, n_des, payload_size):
        def random_actions():
            channels = np.random.randint(0, self.env.n_sub_carrier, size=(self.env.n_veh, self.env.n_des))
            power_selections = np.random.randint(0, len(self.env.V2V_power_dB_list),
                                                 size=(self.env.n_veh, self.env.n_des))
            return np.stack((channels, power_selections), axis=2)

        model_dir = os.path.join(model_root_dir, 'env%d_%d' % (n_veh, n_des))
        if not os.path.exists(model_dir):
            print("目录不存在")
            return
        for i in range(n_veh):
            for j in range(n_des):
                path = os.path.join(model_dir, "agent%d_%d.m" % (i, j))
                if not os.path.exists(path):
                    print("模型文件不存在")
                    return
                agent = Agent.Agent(i, j, n_veh, self.D)
                agent.read_net_state(torch.load(path))
                self.agents.append(agent)
        steps = int(self.T / self.agent_time_step)
        V2I_sums = []
        p_results = []
        for t_id in range(cycles):
            self.env = Environment.Environment(n_veh, n_des, payload_size)
            V2I_capacity_sum = np.zeros(self.env.n_veh)
            V2V_capacity_sum = np.zeros((self.env.n_veh, self.env.n_des))
            p_sum = 0
            for e in range(rounds):
                print('%d,%d' % (t_id, e))
                self.env.renew_environment()
                self.env.update_small_fading()
                self.env.reset_payload(payload_size)
                self.env.reset_time(self.T)
                all_actions = random_actions()
                for t in range(steps):
                    global_obs = self.env.get_observations(all_actions)
                    for agent in self.agents:
                        local_obs = self.env.get_local_observation(global_obs, agent.veh_index, agent.des_index)
                        action = agent.act(local_obs, 0, e)
                        all_actions[agent.veh_index][agent.des_index] = action
                    V2I_capacity = self.env.get_V2I_capacity(all_actions)
                    V2V_capacity = self.env.get_V2V_capacity(all_actions, True)
                    V2I_capacity_sum += V2I_capacity / (1000 * 1000)
                    V2V_capacity_sum += V2V_capacity
                    global_reward = self.compute_reward(V2I_capacity, V2V_capacity, 0.1, 0.9,
                                                        10)
                    self.env.update_small_fading()
                p_sum += np.count_nonzero(self.env.remain_payload <= 0) / (self.env.n_veh * self.env.n_des)
            V2I_sum_avg = np.sum(V2I_capacity_sum) / (rounds * steps)
            p_avg = p_sum / rounds
            print(V2I_sum_avg, p_avg)
            V2I_sums.append(V2I_sum_avg)
            p_results.append(p_avg)
        self.logger.info(
            'B = %d   total rounds = %d   V2I sum capacity = %fMbps  V2V transmission probability = %f' % (
                payload_size, rounds, np.mean(V2I_sums), np.mean(p_results)))

    def save_model(self, model_root_dir):
        model_dir = os.path.join(model_root_dir, 'env%d_%d/' % (self.env.n_veh, self.env.n_des))
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        for agent in self.agents:
            model = agent.get_net_state()
            path = model_dir + "agent%d_%d.m" % (agent.veh_index, agent.des_index)
            torch.save(model, path)

    def show(self):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set(title='MARL-loss', ylabel='loss', xlabel='episodes')
        episodes = range(0, 3000)
        ax.plot(episodes, self.loss_list)
        plt.show()

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set(title='MARL-reward', ylabel='reward', xlabel='episodes')
        ax.plot(episodes, self.reward_list)
        plt.show()

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set(title='MARL-V2I', ylabel='V2I_rate(MBps)', xlabel='episodes')
        ax.plot(episodes, self.V2I_rate_list)
        plt.show()

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set(title='MARL-V2V', ylabel='V2V payload transmission probability', xlabel='episodes')
        ax.plot(episodes, self.V2I_rate_list)
        plt.show()


def get_epsilon(x):
    if x < 2400:
        return 1 - 0.98 / 2399 * x
    else:
        return 0.02


if __name__ == '__main__':
    log_path = './log'
    log_path = os.path.join(log_path, '2023-5-18.log')
    algorithm = Algorithm(log_path)
    algorithm.n_episodes = 3000
    algorithm.train(4, 1, 2 * 1060, 0.1, 0.9, success_reward=10)
    algorithm.save_model('./model v4.1')
    algorithm.show()
    # algorithm.test('./new_models 5-17 beta=10/', 10, 1000, 4, 1, 2 * 1060)
