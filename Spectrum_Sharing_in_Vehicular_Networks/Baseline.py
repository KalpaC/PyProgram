# Baseline 2023/5/10 18:37
import Environment
import numpy as np
import time


class RandomBaseline:
    def __init__(self):
        self.B = 4 * 1060
        self.env = Environment.Environment(4, 1, self.B)
        self.T = 0.1
        self.time_step = 0.001
        pass

    def random_actions(self):
        channels = np.random.randint(0, self.env.n_sub_carrier, size=(self.env.n_veh, self.env.n_des))
        power_selections = np.random.randint(0, len(self.env.V2V_power_dB_list),
                                             size=(self.env.n_veh, self.env.n_des))
        return np.stack((channels, power_selections), axis=2)

    def determined_action(self):
        channels = np.array([0,1,2,3]).reshape((4,1))
        power_selections = np.array([0,0,0,0]).reshape((4,1))
        return np.stack((channels, power_selections), axis=2)

    def run(self, cycles, rounds, payload_size):
        maximum = 0
        self.B = payload_size
        steps = int(self.T / self.time_step)
        V2I_sums = []
        p_results = []
        V2V_record = []
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
                    all_actions = self.random_actions()
                    # all_actions = self.determined_action()
                    V2I_capacity = self.env.get_V2I_capacity(all_actions)
                    V2I_capacity_sum += V2I_capacity / (1000 * 1000)
                    V2V_capacity = self.env.get_V2V_capacity(all_actions, True)
                    s = max([V2V_capacity[i][j]
                             for i in range(self.env.n_veh)
                             for j in range(self.env.n_des)])
                    if s > maximum:
                        maximum = s
                    V2V_record.append(V2V_capacity.mean())
                    self.env.update_small_fading()
                p_sum += np.count_nonzero(self.env.remain_payload <= 0) / (self.env.n_veh * self.env.n_des)
            V2I_sum_avg = np.sum(V2I_capacity_sum) / (rounds * steps)
            p_avg = p_sum / rounds
            print(V2I_sum_avg, p_avg)
            V2I_sums.append(V2I_sum_avg)
            p_results.append(p_avg)
        print("V2I信道容量和为",np.mean(V2I_sums),"V2V平均传输成功率为", np.mean(p_results))
        print("Random baseline 最大V2V容量为%dbps" % maximum)
        print("平均V2V信道容量为",np.mean(V2V_record))


if __name__ == "__main__":
    rb = RandomBaseline()
    rb.run(100, 1000, 2 * 1060)
