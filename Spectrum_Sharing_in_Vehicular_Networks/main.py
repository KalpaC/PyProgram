# process2 2023/5/15 14:04

# main 2023/5/15 13:27

import os

import numpy as np
import Environment
import Agent
import torch
import logging


class Algorithm:
    def __init__(self, logger_path):
        self.agents = []
        self.env = None
        self.T = 0.1
        self.agent_time_step = 0.001
        self.D = 1000000
        self.n_episodes = 3000
        self.init_logger(logger_path)

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
        self.env = Environment.Environment(n_veh, n_des, payload_size)

        def random_actions():
            channels = np.random.randint(0, self.env.n_sub_carrier, size=(self.env.n_veh, self.env.n_des))
            power_selections = np.random.randint(0, len(self.env.V2V_power_dB_list),
                                                 size=(self.env.n_veh, self.env.n_des))
            return np.stack((channels, power_selections), axis=2)

        for i in range(n_veh):
            for j in range(n_des):
                self.agents.append(Agent.Agent(i, j, self.env, self.D))
        steps = int(self.T / self.agent_time_step)
        for e in range(self.n_episodes):
            # 计算epsilon
            epsilon = get_epsilon(e)
            self.env.renew_environment()
            self.env.update_small_fading()
            self.env.reset_payload(payload_size)
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
                global_reward = self.compute_reward(V2I_capacity, V2V_capacity, V2I_weight, V2V_weight, success_reward)
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
                   (np.count_nonzero(self.env.remain_payload > 0) / (self.env.n_veh * self.env.n_des)) + "  " \
                   + "平均奖励值：%fMBps" % (reward_sum / steps)
            self.logger.info(info)
            for agent in self.agents:
                agent.train()

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

        model_dir = model_root_dir + 'env%d_%d/' % (n_veh, n_des)
        if not os.path.exists(model_dir):
            print("目录不存在")
            return
        for i in range(n_veh):
            for j in range(n_des):
                path = model_dir + "agent%d_%d.m" % (i, j)
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
                        action = agent.act(local_obs, 1, e)
                        all_actions[agent.veh_index][agent.des_index] = action
                    V2I_capacity_sum += self.env.get_V2I_capacity(all_actions) / (1000 * 1000)
                    V2V_capacity_sum += self.env.get_V2V_capacity(all_actions, True)
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


def get_epsilon(x):
    if x < 2400:
        return 1 - 0.98 / 2399 * x
    else:
        return 0.02


if __name__ == '__main__':
    algorithm = Algorithm('测试')
    algorithm.n_episodes = 3000
    # algorithm.train()
    # algorithm.save_model('./models 5-15 beta=20/')
    algorithm.test('./models 5-15 beta=20/', 10, 1000, 4, 1, 5 * 1060)
