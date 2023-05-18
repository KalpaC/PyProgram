# Baseline 2023/5/10 18:37
import Environment
import numpy as np
import time


class V2IUpperBound:
    def __init__(self):
        self.B = 4 * 1060
        self.env = Environment.Environment(4, 1, self.B)
        self.T = 0.1
        self.time_step = 0.001
        pass

    def close_V2V(self):
        channels = np.zeros((self.env.n_veh, self.env.n_des),dtype='int32')
        power_selections = np.full((self.env.n_veh, self.env.n_des), len(self.env.V2V_power_dB_list) - 1,dtype='int32')
        return np.stack((channels, power_selections), axis=2)

    def run(self, cycles, rounds, payload_size):
        maximum = 0
        self.B = payload_size
        steps = int(self.T / self.time_step)
        V2I_sums = []
        p_results = []
        for t_id in range(cycles):
            V2I_capacity_sum = np.zeros(self.env.n_veh)
            p_sum = 0.
            for e in range(rounds):
                print('%d,%d' % (t_id, e))
                self.env.renew_environment()
                self.env.update_small_fading()
                self.env.reset_payload(self.B)
                self.env.reset_time(self.T)
                for t in range(steps):
                    all_actions = self.close_V2V()
                    V2I_capacity = self.env.get_V2I_capacity(all_actions)
                    V2I_capacity_sum += V2I_capacity / (1000 * 1000)
                    # print("episode-%d, step-%d, V2I_capacity（Mbps）" % (e, t), (V2I_capacity / 1000000), "sum:",
                    #       np.sum((V2I_capacity / 1000000)))
                    V2V_capacity = self.env.get_V2V_capacity(all_actions, True)
                    s = max([V2V_capacity[i][j]
                             for i in range(self.env.n_veh)
                             for j in range(self.env.n_des)])
                    if s > maximum:
                        maximum = s
                    self.env.update_small_fading()
                p_sum += np.count_nonzero(self.env.remain_payload <= 0) / (self.env.n_veh * self.env.n_des)
            #     print("V2V交付成功占比：",
            #           sum([1 if payload <= 0 else 0 for payload in self.env.remain_payload.reshape(
            #               (self.env.n_veh * self.env.n_des))]) / (self.env.n_veh * self.env.n_des))
            V2I_sum_avg = np.sum(V2I_capacity_sum) / (rounds * steps)
            p_avg = p_sum / rounds
            print(V2I_sum_avg, p_avg)
            V2I_sums.append(V2I_sum_avg)
            p_results.append(p_avg)
        print(np.mean(V2I_sums), np.mean(p_results))
        print("Random baseline 最大V2V容量为%dbps" % maximum)


if __name__ == "__main__":
    ub = V2IUpperBound()
    ub.run(10, 1000, 3 * 1060)
