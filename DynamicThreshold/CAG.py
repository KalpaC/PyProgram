# CIMFS 2023/3/20 19:29
import numpy as np
from PyEMD import CEEMDAN
from matplotlib import pyplot as plt


class CAG:
    def __init__(self, dataLayer):
        self.dataLayer = dataLayer

    def getCIMFs(self, column: str):
        seq = np.array(self.dataLayer.getResampleDataFrame()[column])
        ceemdan = CEEMDAN()
        ceemdan.ceemdan(seq)
        cIMFs, res = ceemdan.get_imfs_and_residue()
        # 图形展示
        plt.figure(figsize=(12, 9))
        plt.subplots_adjust(hspace=0.1)
        plt.subplot(cIMFs.shape[0] + 2, 1, 1)
        plt.plot(seq, 'r')
        plt.ylabel('OriginData')
        for i in range(cIMFs.shape[0]):
            plt.subplot(cIMFs.shape[0] + 3, 1, i + 3)
            plt.plot(cIMFs[i], 'g')
            plt.ylabel("IMF %i" % (i + 1))
            plt.locator_params(axis='x', nbins=10)
        plt.subplot(cIMFs.shape[0] + 3, 1, cIMFs.shape[0] + 3)
        plt.plot(res, 'g')
        plt.ylabel('Residual')
        plt.show()
        return cIMFs, res