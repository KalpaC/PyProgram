# TimeSequence 2023/1/4 21:46
import scipy.interpolate as spi
import numpy as np
from PyEMD import CEEMDAN


class TimeSequence:
    def __init__(self, timeSeq: list = None, startStamp=0):
        if timeSeq:
            self.timeSeq = timeSeq
        else:
            self.timeSeq = []
        self.startStamp = startStamp


    def getCIMFs(self, numOfWhiteNoise:int, amplitude:float, limit:float):
        ceemdan = CEEMDAN(numOfWhiteNoise,amplitude)
        ceemdan.ceemdan(np.array(self.timeSeq))
        cIMFs, res = ceemdan.get_imfs_and_residue()
        return cIMFs, res

