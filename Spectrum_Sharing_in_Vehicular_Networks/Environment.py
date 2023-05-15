# 作者 Ajex
# 创建时间 2023/5/5 19:11
# 文件名 Environment.py
import random
import time
import numpy as np
import math

from typing import List

def kmph2mps(velocity):
    return velocity * 1000 / 3600

def power_dB2W(power_dB):
    return 10 ** (power_dB / 10)

class Vehicle:
    # Vehicle simulator: include all the information for a vehicle
    def __init__(self, start_position, start_direction, velocity):
        self.position = start_position
        self.direction = start_direction
        self.velocity = velocity
        self.destinations = []
        self.turn = ''


class V2VChannels:
    def __init__(self, n_veh, n_sub_carrier):
        self.t = 0
        self.height_BS = 25  # m
        self.height_veh = 1.5  # m
        self.carrier_frequency = 2  # GHz
        self.decorrelation_distance = 10  # m
        self.shadow_std = 3
        self.n_veh = n_veh
        self.n_sub_carrier = n_sub_carrier
        self.positions = None
        self.last_distance = None
        self.distance = np.zeros(shape=(self.n_veh, self.n_veh))  # 二维矩阵
        self.PathLoss = np.zeros(shape=(self.n_veh, self.n_veh))
        self.Shadow = None
        self.FastFading = None

    def update_positions(self, positions):
        self.positions = np.array(positions)
        self.last_distance = np.copy(self.distance)
        for i in range(self.n_veh):
            for j in range(self.n_veh):
                # np.linalg.norm用于计算向量的模，此处可以用于计算两点间距离
                self.distance[i][j] = np.linalg.norm(self.positions[i] - self.positions[j])

    def update_pathLoss(self):
        for i in range(self.n_veh):
            for j in range(self.n_veh):
                self.PathLoss[i][j] = self.get_pathLoss(i, j)

    def update_shadow(self):
        if self.Shadow is None:
            self.Shadow = np.random.normal(0, self.shadow_std, size=(self.n_veh, self.n_veh))
        else:
            delta_distance = np.abs(self.distance - self.last_distance)
            self.Shadow = np.exp(-1 * (delta_distance / self.decorrelation_distance)) * self.Shadow + np.sqrt(
                1 - np.exp(-2 * (delta_distance / self.decorrelation_distance))) * np.random.normal(0, self.shadow_std,
                                                                                                    size=(self.n_veh,
                                                                                                          self.n_veh))

    def get_pathLoss(self, i, j):
        d1 = abs(self.positions[i][0] - self.positions[j][0])
        d2 = abs(self.positions[i][1] - self.positions[j][1])
        d = math.hypot(d1, d2) + 0.001
        d_breakpoint = 4 * (self.height_BS - 1) * (self.height_veh - 1) * self.carrier_frequency * (10 ** 9) / (
                3 * 10 ** 8)

        def PL_Los(d):
            if d <= 3:
                return 22.7 * np.log10(3) + 41 + 20 * np.log10(self.carrier_frequency / 5)
            else:
                if d < d_breakpoint:
                    return 22.7 * np.log10(d) + 41 + 20 * np.log10(self.carrier_frequency / 5)
                else:
                    return 40.0 * np.log10(d) + 9.45 - 17.3 * np.log10(self.height_BS) - 17.3 * np.log10(
                        self.height_veh) + 2.7 * np.log10(self.carrier_frequency / 5)

        def PL_NLos(d_a, d_b):
            n_j = max(2.8 - 0.0024 * d_b, 1.84)
            return PL_Los(d_a) + 20 - 12.5 * n_j + 10 * n_j * np.log10(d_b) + 3 * np.log10(self.carrier_frequency / 5)

        if min(d1, d2) < 7:
            PL = PL_Los(d)
            # self.ifLOS = True
            self.shadow_std = 3
        else:
            PL = min(PL_NLos(d1, d2), PL_NLos(d2, d1))
            # self.ifLOS = False
            self.shadow_std = 4  # if Non line of sight, the std is 4
        return PL

    def update_fast_fading(self):
        h = 1 / np.sqrt(2) * (
                np.random.normal(size=(self.n_veh, self.n_veh, self.n_sub_carrier)) + 1j * np.random.normal(
            size=(self.n_veh, self.n_veh, self.n_sub_carrier)))
        self.FastFading = 20 * np.log10(np.abs(h))


class V2IChannels:
    #
    def __init__(self, n_veh, n_sub_carrier):
        self.n_veh = n_veh
        self.BS_position = np.array([750 / 2, 1299 / 2])  # Suppose the BS is in the center
        self.n_sub_carrier = n_sub_carrier
        self.height_BS = 25  # m
        self.height_veh = 1.5  # m
        self.shadow_std = 8  # dB
        self.Decorrelation_distance = 50  # m
        self.Shadow = None
        self.positions = None
        self.last_distance = None
        self.distance_to_BS = None
        self.PathLoss = None

    def update_positions(self, positions):
        # 传入的是按序的车辆位置信息，即position[0]对应的vehicle[1].position
        if len(positions) != self.n_veh:
            print("错误的代码，位置信息应与车辆数对应")
        self.positions = np.array(positions)
        self.last_distance = self.distance_to_BS
        new_delta_horizon = np.linalg.norm(self.positions - self.BS_position, axis=1)
        self.distance_to_BS = np.hypot(new_delta_horizon, (self.height_BS - self.height_veh))

    def update_pathLoss(self):
        self.PathLoss = 128.1 + 37.6 * np.log10(self.distance_to_BS / 1000)

    def update_shadow(self):
        """
        更新shadow，或初始化。
        更新时，计算所有车辆原位置与基站的三维距离和新位置与基站的三维距离，而后做差作为距离变化值。
        """
        if self.last_distance is None:
            self.Shadow = np.random.normal(0, self.shadow_std, self.n_veh)
            return
        delta_distance = np.abs(self.distance_to_BS - self.last_distance)
        self.Shadow = np.exp(-1 * (delta_distance / self.Decorrelation_distance)) * self.Shadow + \
                      np.random.normal(0, self.shadow_std, self.n_veh) * \
                      np.sqrt(1 - np.exp(-2 * (delta_distance / self.Decorrelation_distance)))

    def update_fast_fading(self):
        # 每1ms更新一次
        h = 1 / np.sqrt(2) * (np.random.normal(size=(self.n_veh, self.n_sub_carrier)) + 1j * np.random.normal(
            size=(self.n_veh, self.n_sub_carrier)))
        self.FastFading = 20 * np.log10(np.abs(h))


class Environment:
    up_lanes = [3.5 / 2, 3.5 / 2 + 3.5, 250 + 3.5 / 2, 250 + 3.5 + 3.5 / 2, 500 + 3.5 / 2, 500 + 3.5 + 3.5 / 2]
    down_lanes = [250 - 3.5 - 3.5 / 2, 250 - 3.5 / 2, 500 - 3.5 - 3.5 / 2, 500 - 3.5 / 2, 750 - 3.5 - 3.5 / 2,
                  750 - 3.5 / 2]
    left_lanes = [3.5 / 2, 3.5 / 2 + 3.5, 433 + 3.5 / 2, 433 + 3.5 + 3.5 / 2, 866 + 3.5 / 2, 866 + 3.5 + 3.5 / 2]
    right_lanes = [433 - 3.5 - 3.5 / 2, 433 - 3.5 / 2, 866 - 3.5 - 3.5 / 2, 866 - 3.5 / 2, 1299 - 3.5 - 3.5 / 2,
                   1299 - 3.5 / 2]
    lanes = {'up': up_lanes, 'down': down_lanes, 'left': left_lanes, 'right': right_lanes}
    width = 750
    height = 1299

    def __init__(self, M, K, B):
        """
        尊重论文符号的参数列表，以及使用符合论文设定的默认环境参数值。
        :param M: 在本环境下等于子载波数、车辆数、V2I link数
        :param K: 每辆车的“邻居”个数，即每辆车与多少辆其他车建立V2V link
        :param B: 场景初始时所有V2V link的有效载荷数量
        :return: None
        """
        # 固定的环境参数或变量
        self.vehNoiseFigure = 9  # dB
        self.bandwidth = 4 * 1000000  # MHz
        self.V2I_power_dB = 23
        self.bsNoiseFigure = 5  # dB
        self.bsAntGain = 8  # dB
        self.vehAntGain = 3  # dB
        sig2_dB = -114  # dB
        self.sig2 = power_dB2W(sig2_dB)
        self.position_time_step = 0.1
        self.action_time_step = 0.001  # s
        self.vehicles: List[Vehicle] = []
        self.remain_time = 0.1  # s

        # 与V2V有关而可能会改变的部分
        self.n_veh = M
        self.n_sub_carrier = M
        self.n_des = K
        self.payload_size = B  # bytes
        # self.beta = 8 * 1000 * 1000  # 尚未确定
        self.V2V_power_dB_list = [23, 10, 5, -100]
        self.init_all_vehicles()
        self.V2IChannels = V2IChannels(self.n_veh, self.n_sub_carrier)
        self.V2VChannels = V2VChannels(self.n_veh, self.n_sub_carrier)
        self.V2I_channels_with_fastfading = None
        self.V2V_channels_with_fastfading = None
        self.remain_payload = np.full((self.n_veh, self.n_des), self.payload_size)

    def reset_payload(self, B):
        self.payload_size = B
        self.remain_payload = np.full((self.n_veh, self.n_des), self.payload_size)

    def reset_time(self, T):
        self.remain_time = T

    def print_positions(self):
        print([(v.position, v.direction, v.turn) for v in self.vehicles])

    def update_large_fading(self, positions, time_step):
        self.V2IChannels.update_positions(positions)
        self.V2VChannels.update_positions(positions)
        self.V2IChannels.update_pathLoss()
        self.V2VChannels.update_pathLoss()
        self.V2IChannels.update_shadow()
        self.V2VChannels.update_shadow()

    def update_small_fading(self):
        self.V2IChannels.update_fast_fading()
        self.V2VChannels.update_fast_fading()
        self.compute_channels_with_fastfading()

    def test(self):
        n_veh = 4
        n_sub_carrier = 4
        self.init_all_vehicles()
        positions = [c.position for c in self.vehicles]
        v2i = V2IChannels(n_veh, n_sub_carrier)
        v2i.update_positions(positions)
        v2i.update_pathLoss()
        v2i.update_shadow()
        v2i.update_fast_fading()
        # self.print_positions()
        for i in range(int(1000 / self.position_time_step)):
            print("\nstep=%d:" % i)
            self.renew_positions()
            v2i.update_positions(positions)
            v2i.update_pathLoss()
            v2i.update_shadow()
            v2i.update_fast_fading()
            self.print_positions()
            time.sleep(5)

    def add_new_vehicle(self):
        # 目前所有车辆的速度均为36km/h，如果需要修改或引入随机性，请重新设置
        direction = np.random.choice(['up', 'down', 'left', 'right'])
        road = np.random.randint(0, len(self.lanes[direction]))
        if direction == 'up' or direction == 'down':
            x = self.lanes[direction][road]
            y = np.random.randint(3.5 / 2, self.height - 3.5 / 2)
        else:
            x = np.random.randint(3.5 / 2, self.width - 3.5 / 2)
            y = self.lanes[direction][road]
        position = [x, y]
        self.vehicles.append(Vehicle(position, direction, np.random.randint(15,60)))

    def init_all_vehicles(self):
        # 初始化全部的vehicle，
        for i in range(self.n_veh):
            self.add_new_vehicle()
        self.get_destination()

    def get_destination(self):
        # 找到对每辆车找到距离它最近的self.n_des辆车
        # 每次更新位置之后都需要重新判断，因为数据包的有效期恰好也过了
        positions = np.array([c.position for c in self.vehicles])
        distance = np.zeros((self.n_veh, self.n_veh))
        for i in range(self.n_veh):
            for j in range(self.n_veh):
                # np.linalg.norm用于计算向量的模，此处可以用于计算两点间距离
                distance[i][j] = np.linalg.norm(positions[i] - positions[j])
        for i in range(self.n_veh):
            sort_idx = np.argsort(distance[:, i])
            self.vehicles[i].destinations = sort_idx[1:1 + self.n_des]

    def renew_environment(self):
        # 执行周期维100ms的统一更新，所以并不包括FastFading的更新
        self.renew_positions()
        self.get_destination()
        positions = [c.position for c in self.vehicles]
        self.V2IChannels.update_positions(positions)
        self.V2VChannels.update_positions(positions)
        self.V2IChannels.update_pathLoss()
        self.V2VChannels.update_pathLoss()
        self.V2IChannels.update_shadow()
        self.V2VChannels.update_shadow()
        self.V2V_channels_abs = self.V2VChannels.PathLoss + self.V2VChannels.Shadow + 50 * np.identity(
            len(self.vehicles))
        self.V2I_channels_abs = self.V2IChannels.PathLoss + self.V2IChannels.Shadow

    def compute_channels_with_fastfading(self):
        # 该函数只负责更新信道衰落，并不负责刷新信道
        V2V_channels_with_fastfading = np.repeat(self.V2V_channels_abs[:, :, np.newaxis], self.n_sub_carrier, axis=2)
        self.V2V_channels_with_fastfading = V2V_channels_with_fastfading - self.V2VChannels.FastFading
        V2I_channels_with_fastfading = np.repeat(self.V2I_channels_abs[:, np.newaxis], self.n_sub_carrier, axis=1)
        self.V2I_channels_with_fastfading = V2I_channels_with_fastfading - self.V2IChannels.FastFading
        self.remain_time -= 0.001

    def compute_V2V_interference(self, all_actions):
        V2V_interference = np.zeros((self.n_veh, self.n_des, self.n_sub_carrier))
        channel = all_actions[:, :, 0]
        for m in range(self.n_sub_carrier):
            indexes = np.argwhere(channel == m)
            for i, j in indexes:
                V2V_interference[i][j][m] = power_dB2W(
                    self.V2I_power_dB
                    + self.interference_dB_from_V2I_to_V2V((i, j), m)
                )
                for other_i, other_j in indexes:
                    if other_i != i and other_j != j:
                        if all_actions[other_i][other_j][1] == len(self.V2V_power_dB_list) - 1:
                            continue
                        power_dB = self.V2V_power_dB_list[all_actions[other_i][other_j][1]]
                        # 注意other_i是另一辆车的车号，other_j是des号，所以用other_i来定位V2V信道的发射位置
                        # 所以发送位置下标是other_i，接收位置下标是i的destination[j]
                        V2V_interference[i][j][m] += power_dB2W(
                            power_dB
                            + self.interference_dB_from_other_V2V((i, j), (other_i, other_j), m)
                        )
        return V2V_interference

    def compute_V2I_interference(self, all_actions):
        # V2I_channel_with_fastfading的本质上是一辆车的位置发送的信号到基站在m子载波上的信道增益
        # 而V2V k到基站的信号干扰增益本质上也是车到基站在信号在m子载波上的信道增益
        # 同理，V2V_channel_with_fastfading本质上是车A位置向车B的位置发送的信号在子载波m上的信道增益
        V2I_interference = np.zeros(self.n_sub_carrier)
        for i in range(self.n_veh):
            for j in range(self.n_des):
                if all_actions[i][j][0] == len(self.V2V_power_dB_list) - 1:
                    continue
                m = all_actions[i][j][0]
                V2I_interference[m] += power_dB2W(
                    self.V2V_power_dB_list[all_actions[i][j][1]]
                    + self.interference_dB_from_V2V_to_BS((i, j), m)
                )
        return V2I_interference

    def get_observations(self, all_actions):
        payloads = self.remain_payload
        remain_time = self.remain_time
        V2V_interference = self.compute_V2V_interference(all_actions)
        gains = np.zeros((self.n_veh, self.n_des, self.n_sub_carrier, 4), dtype='float64')
        for i in range(self.n_veh):
            for j in range(self.n_des):
                channel = (i, j)
                for m in range(self.n_sub_carrier):
                    gains[i, j, m][0] = self.gain_dB_of_V2V(channel, m)
                    for other_i in range(self.n_veh):
                        for other_j in range(self.n_des):
                            other_channel = (other_i, other_j)
                            if channel != other_channel:
                                gains[i, j, m][1] += self.interference_dB_from_other_V2V(channel, other_channel, m)
                    gains[i, j, m][2] = self.interference_dB_from_V2V_to_BS(channel, m)
                    gains[i, j, m][3] = self.interference_dB_from_V2I_to_V2V(channel, m)
        return payloads, remain_time, V2V_interference, gains

    def get_local_observation(self, global_obs, veh_idx, des_idx):
        """
        基于get_states函数的输出的拆分函数
        :param global_obs: get_states函数的输出
        :param veh_idx: Agent对应的车辆号
        :param des_idx: Agent对应的des号
        :return: Agent k的local state
        """
        return np.full(1, global_obs[0][veh_idx][des_idx]), np.full(1, global_obs[1]), global_obs[2][veh_idx][des_idx], \
               global_obs[3][veh_idx][des_idx]

    def gain_dB_of_V2V(self, V2V_Channel, m):
        """
        g_k[m] V2V link k的发射器到接收器之间的信道增益dB
        :param V2V_Channel: (veh_id, des_id)
        :param m: 子载波编号m
        :return: 信道增益dB，正值则为增益
        """
        sent = V2V_Channel[0]
        recv = self.vehicles[sent].destinations[V2V_Channel[1]]
        minus = self.V2V_channels_with_fastfading[sent][recv][m] + self.vehNoiseFigure
        plus = self.vehAntGain * 2
        return -minus + plus

    def gain_dB_of_V2I(self, m):
        """
        g_{m,B}[m]
        :param m: V2I link m，子载波编号m（该环境默认二者相同）
        :return: 信道增益dB，正值则为增益
        """
        sent = m
        minus = self.V2I_channels_with_fastfading[m][m] + self.bsNoiseFigure
        plus = self.vehAntGain + self.bsAntGain
        return -minus + plus

    def interference_dB_from_other_V2V(self, V2V_Channel, other_V2V, m):
        """
        g_{k',k}[m] 从其他V2V k'的发射器到本V2V k的接收器的干扰功率增益dB
        :param V2V_Channel: (veh_id, des_id)
        :param other_V2V: (veh_id, des_id)
        :param m: 子载波编号m
        :return: 信道增益dB，正值则为增益
        """
        sent = other_V2V[0]
        recv = self.vehicles[V2V_Channel[0]].destinations[V2V_Channel[1]]
        minus = self.V2V_channels_with_fastfading[sent][recv][m] + self.vehNoiseFigure
        plus = self.vehAntGain * 2
        return -minus + plus

    def interference_dB_from_V2V_to_BS(self, V2V_Channel, m):
        """
        g_{k,B}[m] 从V2V link k的发射器到基站B之间的干扰功率增益dB
        :param V2V_Channel: (veh_id, des_id)
        :param m: 子载波编号m
        :return: 信道增益dB，正值则为增益
        """
        sent = V2V_Channel[0]
        minus = self.V2I_channels_with_fastfading[sent][m] + self.bsNoiseFigure
        plus = self.vehAntGain + self.bsAntGain
        return -minus + plus

    def interference_dB_from_V2I_to_V2V(self, V2V_Channel, m):
        """
        \\hat{g}_{m,k}[m] 从V2I link的发射器m到V2V link k的接收器之间的干扰功率增益dB
        :param V2V_Channel: (veh_id, des_id)
        :param m: 子载波编号m，也是V2I link m
        :return: 信道增益dB，正值则为增益
        """
        sent = m
        recv = self.vehicles[V2V_Channel[0]].destinations[V2V_Channel[1]]
        minus = self.V2V_channels_with_fastfading[sent][recv][m] + self.vehNoiseFigure
        plus = self.vehAntGain * 2
        return -minus + plus

    def get_V2I_capacity(self, all_actions):
        V2I_interference = self.compute_V2I_interference(all_actions)
        V2I_Signals = power_dB2W(
            self.V2I_power_dB
            + np.array([self.gain_dB_of_V2I(m) for m in range(self.n_sub_carrier)])
        )
        V2I_SINR = np.divide(V2I_Signals, self.sig2 + V2I_interference)
        V2I_Capacity = self.bandwidth * np.log2(1 + V2I_SINR)
        return V2I_Capacity

    def get_V2V_capacity(self, all_actions, update_payload=True):
        V2V_interference = self.compute_V2V_interference(all_actions)
        V2V_capacity = np.zeros((self.n_veh, self.n_des))
        for i in range(self.n_veh):
            for j in range(self.n_des):
                # 计算信道m的容量
                m = all_actions[i][j][0]
                if all_actions[i][j][1] == len(self.V2V_power_dB_list) - 1:
                    continue
                power = self.V2V_power_dB_list[all_actions[i][j][1]]
                signal = power_dB2W(
                    power
                    + self.gain_dB_of_V2V((i, j), m)
                )
                noise_plus_interference = self.sig2 + V2V_interference[i][j][m]
                V2V_capacity[i][j] = self.bandwidth * np.log2(1 + signal / noise_plus_interference)
                if update_payload:
                    self.remain_payload[i][j] -= np.floor(V2V_capacity[i][j] * self.action_time_step / 8)
        return V2V_capacity

    def renew_positions(self):
        # 不能直行的条件判断
        def cant_go_straight(car: Vehicle):
            if car.direction == 'up' and car.position[1] > self.left_lanes[-1]:
                return True
            elif car.direction == 'down' and car.position[1] < self.right_lanes[0]:
                return True
            elif car.direction == 'left' and car.position[0] < self.down_lanes[0]:
                return True
            elif car.direction == 'right' and car.position[0] > self.up_lanes[-1]:
                return True
            return False

        # 不能左转的条件判断
        def cant_turn_left(car: Vehicle):
            if cant_go_straight(car):
                return True
            elif car.direction == 'up' and (car.position[0] == self.up_lanes[0] or car.position[0] == self.up_lanes[1]):
                return True
            elif car.direction == 'down' and (
                    car.position[0] == self.down_lanes[-2] or car.position[0] == self.down_lanes[-1]):
                return True
            elif car.direction == 'left' and (
                    car.position[1] == self.left_lanes[0] or car.position[1] == self.left_lanes[1]):
                return True
            elif car.direction == 'right' and (
                    car.position[1] == self.right_lanes[-2] or car.position[1] == self.right_lanes[-1]):
                return True
            return False

        # 对每辆车
        for v in self.vehicles:
            if v.turn == '':
                r = random.uniform(0, 1)
                if cant_go_straight(v) and cant_turn_left(v):
                    v.turn = 'right'
                elif cant_turn_left(v):
                    if 0 <= r < 0.5:
                        v.turn = 'straight'
                    else:
                        v.turn = 'right'
                else:
                    if 0 <= r < 0.25:
                        v.turn = 'left'
                    elif 0.25 <= r < 0.5:
                        v.turn = 'right'
                    else:
                        v.turn = 'straight'
            # 计算出下一步的位置
            delta_distance = self.position_time_step * kmph2mps(v.velocity)
            turn_case = False
            if v.direction == 'up':
                if v.turn == 'right':
                    for right in self.right_lanes:
                        if v.position[1] < right < v.position[1] + delta_distance:
                            turn_case = True
                            v.direction = 'right'
                            exceed = v.position[1] + delta_distance - right
                            v.position[0] += exceed
                            v.position[1] = right
                            break
                else:
                    for left in self.left_lanes:
                        if v.position[1] < left < v.position[1] + delta_distance:
                            turn_case = True
                            if v.turn == 'straight':
                                v.position[1] += delta_distance
                                break
                            v.direction = 'left'
                            exceed = v.position[1] + delta_distance - left
                            v.position[0] -= exceed
                            v.position[1] = left
                            break
                if not turn_case:
                    v.position[1] += delta_distance
            elif v.direction == 'down':
                if v.turn == 'right':
                    # 向下+右转=向左
                    for left in self.left_lanes:
                        if v.position[1] > left > v.position[1] - delta_distance:
                            turn_case = True
                            v.direction = 'left'
                            exceed = left - (v.position[1] - delta_distance)
                            v.position[0] -= exceed
                            v.position[1] = left
                            break
                else:
                    # 向下+左转=向右
                    for right in self.right_lanes:
                        if v.position[1] > right > v.position[1] - delta_distance:
                            turn_case = True
                            if v.turn == 'straight':
                                v.position[1] -= delta_distance
                                break
                            v.direction = 'right'
                            exceed = right - (v.position[1] - delta_distance)
                            v.position[0] += exceed
                            v.position[1] = right
                            break
                if not turn_case:
                    v.position[1] -= delta_distance
            elif v.direction == 'left':
                if v.turn == 'right':
                    # 左+右=上
                    for up in self.up_lanes:
                        if v.position[0] - delta_distance < up < v.position[0]:
                            turn_case = True
                            v.direction = 'up'
                            exceed = up - (v.position[0] - delta_distance)
                            v.position[1] += exceed
                            v.position[0] = up
                            break
                else:
                    # 左+左=下
                    for down in self.down_lanes:
                        if v.position[0] - delta_distance < down < v.position[0]:
                            turn_case = True
                            if v.turn == 'straight':
                                v.position[0] -= delta_distance
                                break
                            v.direction = 'down'
                            exceed = down - (v.position[0] - delta_distance)
                            v.position[1] -= exceed
                            v.position[0] = down
                            break
                if not turn_case:
                    v.position[0] -= delta_distance
            else:
                if v.turn == 'right':
                    # 右+右 = 下
                    for down in self.down_lanes:
                        if v.position[0] < down < v.position[0] + delta_distance:
                            turn_case = True
                            v.direction = 'down'
                            exceed = v.position[0] + delta_distance - down
                            v.position[1] -= exceed
                            v.position[0] = down
                            break
                else:
                    # 右+左=上
                    for up in self.up_lanes:
                        if v.position[0] < up < v.position[0] + delta_distance:
                            turn_case = True
                            if v.turn == 'straight':
                                v.position[0] += delta_distance
                                break
                            v.direction = 'up'
                            exceed = v.position[0] + delta_distance - up
                            v.position[1] += exceed
                            v.position[0] = up
                            break
                if not turn_case:
                    v.position[0] += delta_distance
            if turn_case:
                v.turn = ''
        # 目前成功更新了每辆车的位置
        # 是否需要同时修改fast_fading等数据？先不了，还没有考虑如何存储。

    def __repr__(self):
        vehicle_info = [(v.position, v.direction, v.turn) for v in self.vehicles]
        return "Vehicle information:\n" + str(vehicle_info)

#
# if __name__ == '__main__':
#     env = Environment()
#     env.test()
