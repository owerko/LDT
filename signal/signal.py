import numpy as np
import statistics
from scipy.signal import hilbert
from scipy.optimize import least_squares


class Signal:
    def __init__(self, name):
        self.name = name
        self.x = np.array([1.0, 1.0, 1.0])
        self.time = np.zeros(shape=(1, 1), dtype=np.float64)
        self.Amplitude = np.zeros(shape=(1, 1), dtype=np.float64)

    def __str__(self):
        return f'{self.name}'

    def fun(self, x, t, y):
        return x[0] * np.exp(-x[1] * t) + x[2] - y

    def load_data(self, filePath):
        with open(filePath) as f:
            data = np.loadtxt(f, dtype=np.float64)
            t = data[:, 0]
            A = data[:, 1]
            A2 = A - statistics.mean(A)
            loc_max_A2_begin_ind = np.argmax(A2[:int(0.1 * len(t))])
            loc_max_A2_end_ind = np.argmax(A2[int(0.9 * len(t)):]) + len(A2[:int(0.9 * len(t))])
            t2 = t[loc_max_A2_begin_ind:loc_max_A2_end_ind:1]
            self.time = t2 - t2[0]
            self.Amplitude = A2[loc_max_A2_begin_ind:loc_max_A2_end_ind:1]

    def lsq_resutls(self):
        amplitude_envelope = np.abs(hilbert(self.Amplitude))
        res_lsq = least_squares(self.fun, self.x, args=(self.time, amplitude_envelope))
        return res_lsq

    def softL1_results(self):
        amplitude_envelope = np.abs(hilbert(self.Amplitude))
        res_soft_l1 = least_squares(self.fun, self.x, loss='soft_l1', f_scale=0.1, args=(self.time, amplitude_envelope))
        return res_soft_l1

    def huber_results(self):
        amplitude_envelope = np.abs(hilbert(self.Amplitude))
        res_hub = least_squares(self.fun, self.x, loss='huber', f_scale=0.1, args=(self.time, amplitude_envelope))
        return res_hub

