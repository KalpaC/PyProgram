# 作者 Ajex
# 创建时间 2023/5/5 19:11
# 文件名 Environment.py
import random
import time
import numpy as np
import math


# 一、确定道路环境
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
        self.decorrelation_distance = 10
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
        if self.last_distance is None:
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
            # self.shadow_std = 3
        else:
            PL = min(PL_NLos(d1, d2), PL_NLos(d2, d1))
            # self.ifLOS = False
            # self.shadow_std = 4  # if Non line of sight, the std is 4
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
        print(self.PathLoss)

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
        print(self.Shadow)

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

    def __init__(self):
        self.beta = None  # 尚未确定
        self.vehNoiseFigure = 9  # dB
        self.bandwidth = 4  # MHz
        self.V2I_power_dB = 23
        self.bsNoiseFigure = 5  # dB
        self.bsAntGain = 8  # dB
        self.vehAntGain = 3  # dB
        sig2_dB = -114  # dB
        self.sig2 = power_dB2W(sig2_dB)
        self.payload_size = 1060  # bytes


        self.vehicles: list[Vehicle] = []
        self.position_time_step = 0.1
        self.n_veh = 4
        self.n_sub_carrier = 4
        self.n_des = 1
        self.V2V_power_dB_list = [23, 10, 5]
        self.V2IChannels = V2IChannels(self.n_veh, self.n_sub_carrier)
        self.V2VChannels = V2VChannels(self.n_veh, self.n_sub_carrier)
        self.V2I_channels_with_fastfading = None
        self.V2V_channels_with_fastfading = None
        self.remain_payload = np.ones(self.n_veh) * self.payload_size

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

    def test(self):
        n_veh = 4
        n_sub_carrier = 4
        self.init_all_vehicles(4)
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
        self.vehicles.append(Vehicle(position, direction, 36))

    def init_all_vehicles(self, n):
        # 初始化全部的vehicle，
        for i in range(n):
            self.add_new_vehicle()

    def renew_destination(self):
        # 找到对每辆车找到距离它最近的self.n_neighbor辆车
        positions = np.array([c.position for c in self.vehicles])
        distance = np.zeros((self.n_veh, self.n_veh))
        for i in range(self.n_veh):
            for j in range(self.n_veh):
                # np.linalg.norm用于计算向量的模，此处可以用于计算两点间距离
                distance[i][j] = np.linalg.norm(positions[i] - positions[j])
        for i in range(self.n_veh):
            sort_idx = np.argsort(distance[:, i])
            self.vehicles[i].destinations = np.random.choice(sort_idx[1:self.n_veh // 4 + 1], self.n_des,
                                                             replace=False)

    def renew_environment(self):
        # 执行周期维100ms的统一更新，所以并不包括FastFading的更新
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

    def get_reward(self, all_actions, weight_V2I, weight_V2V):
        # all_action 为3维数组，前两维是车号x des号，最后一维是[子载波号m，功率dB]
        channel = all_actions[:, :, 0]
        power_selection = all_actions[:, :, 1]
        V2I_interference = np.zeros(self.n_sub_carrier)
        for i in range(self.n_veh):
            for j in range(self.n_des):
                V2I_interference[channel[i][j]] += power_dB2W(
                    self.V2V_power_dB_list[power_selection[i][j]]
                    - self.V2I_channels_with_fastfading[i, channel[i, j]]
                    + self.vehAntGain + self.bsAntGain
                    - self.bsNoiseFigure
                )

        # V2I_channel_with_fastfading的本质上是一辆车的位置发送的信号到基站在m子载波上的信道增益
        # 而V2V k到基站的信号干扰增益本质上也是车到基站在信号在m子载波上的信道增益
        # 同理，V2V_channel_with_fastfading本质上是车A位置向车B的位置发送的信号在子载波m上的信道增益
        V2I_interference += self.sig2
        V2I_Signals = power_dB2W(
            self.V2I_power_dB
            - np.diag(self.V2I_channels_with_fastfading)
            + self.vehAntGain + self.bsAntGain
            - self.bsNoiseFigure
        )
        V2I_SINR = np.divide(V2I_Signals, V2I_interference)
        V2I_Capacity = self.bandwidth * np.log2(1 + V2I_SINR)
        V2I_Capacity_sum = np.sum(V2I_Capacity)

        # 接下来计算V2V的信道容量
        V2V_interference = np.zeros((self.n_veh, self.n_des, self.n_sub_carrier))

        for m in range(self.n_sub_carrier):
            indexes = np.argwhere(channel == m)
            for i, j in indexes:
                receive = self.vehicles[i].destinations[j]
                V2V_interference[i][j][m] = power_dB2W(
                    self.V2I_power_dB
                    - self.V2V_channels_with_fastfading[m][receive][m]
                    + self.vehAntGain * 2
                    - self.vehNoiseFigure
                )
                for other_i, other_j in indexes:
                    if other_i != i and other_j != j:
                        power_dB = self.V2V_power_dB_list[power_selection[other_i][other_j]]
                        # 注意other_i是另一辆车的车号，other_j是des号，所以用other_i来定位V2V信道的发射位置
                        # 所以发送位置下标是other_i，接收位置下标是i的destination[j]
                        V2V_interference[i][j][m] += power_dB2W(
                            power_dB
                            - self.V2V_channels_with_fastfading[other_i][receive]
                            + self.vehAntGain * 2
                            - self.vehNoiseFigure
                        )

        V2V_reward = 0
        for i in range(self.n_veh):
            for j in range(self.n_des):
                if self.remain_payload[i][j] > 0:
                    V2V_reward += self.beta
                else:
                    # 计算信道m的容量
                    m = channel[i][j]
                    power = self.V2V_power_dB_list[power_selection[i][j]]
                    receive = self.vehicles[i].destinations[j]
                    signal = power_dB2W(
                        power
                        - self.V2V_channels_with_fastfading[i][receive][m]
                        + self.vehAntGain * 2
                        - self.vehNoiseFigure
                    )
                    noise_plus_interference = self.sig2 + V2V_interference[i][j][m]
                    V2V_capacity = self.bandwidth * np.log2(1 + signal / noise_plus_interference)
                    V2V_reward += V2V_capacity
        return weight_V2I * V2I_Capacity_sum + weight_V2V * V2V_reward

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


if __name__ == '__main__':
    env = Environment()
    env.test()
