import numpy as np  # usunięcie importu statistics
from scipy.signal import hilbert 
from scipy.optimize import least_squares


class Signal:
    def __init__(self, name):
        self.name = name
        self.x = np.array([1.0, 1.0, 1.0])
        self.time = np.zeros(shape=(1), dtype=np.float64)  # zminimalizowanie wymiarów wartości domyślnej atrybutu np. array z 1,1 do 1
        self.Amplitude = np.zeros(shape=(1), dtype=np.float64)
        self.amplitude_envelope = np.zeros(shape=(1), dtype=np.float64)  # dodanie atrybutu

    def __str__(self):
        return f'{self.name}'

    def fun(self, x, t, y):
        return x[0] * np.exp(-x[1] * t) + x[2] - y

    def load_data(self, filePath):
        data = np.loadtxt(filePath, dtype=np.float64)  # usunięcie podowojenia w imporcie pliku
        t = data[:, 0]
        A = data[:, 1]
        A2 = A - np.mean(A)  # korzystam z numpaja, skoro i tak musi być
        loc_max_A2_begin_ind = np.argmax(A2[:int(0.1 * len(t))])
        loc_max_A2_end_ind = np.argmax(A2[int(0.9 * len(t)):]) + len(A2[:int(0.9 * len(t))])
        t2 = t[loc_max_A2_begin_ind:loc_max_A2_end_ind:1]
        self.time = t2 - t2[0]
        self.Amplitude = A2[loc_max_A2_begin_ind:loc_max_A2_end_ind:1]
        self.amplitude_envelope = np.abs(hilbert(self.Amplitude))
    
    def compute_sampling_spacing(self): #dodanie nowej fukncji
        """Computes key parameters of sampling features of the input signal from in-situ device"""
        time_diffs = np.diff(self.time[:])
        mean_of_time_diffs = np.mean(time_diffs)
        median_of_time_diffs = np.median(time_diffs)
        std_of_time_diffs = np.std(time_diffs)  # standard deviation 
        return [mean_of_time_diffs, median_of_time_diffs, std_of_time_diffs]

    def compute_period(self):
        """Computes period of structural response based on dft"""
        pass  # do zrobienia

    def lsq_resutls(self):
        res_lsq = least_squares(self.fun, self.x, args=(self.time, self.amplitude_envelope)) # usunięcie genrowania zmiennej lokalnej obwiedni
        return res_lsq

    def softL1_results(self):
        res_soft_l1 = least_squares(self.fun, self.x, loss='soft_l1', f_scale=0.1, args=(self.time, self.amplitude_envelope)) # jw
        return res_soft_l1

    def huber_results(self):
        res_hub = least_squares(self.fun, self.x, loss='huber', f_scale=0.1, args=(self.time, self.amplitude_envelope))  # jw
        return res_hub

