import numpy as np  # usunięcie importu statistics
from scipy.signal import hilbert, butter, lfilter, freqz
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
        self.time_cutted = self.time[int(0.02 * len(self.time)):int(0.98 * len(self.time))] # docinka 
        self.Amplitude_ = A2[loc_max_A2_begin_ind:loc_max_A2_end_ind:1]
    
    def compute_sampling_spacing(self): #dodanie nowej fukncji
        """Computes key parameters of sampling features of the input signal from in-situ device"""
        time_diffs = np.diff(self.time[:])
        mean_of_time_diffs = np.mean(time_diffs)
        median_of_time_diffs = np.median(time_diffs)
        std_of_time_diffs = np.std(time_diffs)  # standard deviation
        self.fs = 1 / median_of_time_diffs  # sample rate [Hz]
        return [mean_of_time_diffs, median_of_time_diffs, std_of_time_diffs]

    def compute_period(self):
        """Computes period features of structural response """
        A_sings = np.sign(self.Amplitude_[:])  # poprawa z uwagi na filtr !!! 
        A_diffs = np.diff(A_sings[:])
        sign_changes_of_A = np.where(A_diffs != 0)[0]
        time_marks_of_roots = self.time[sign_changes_of_A]  # stemple czasowe, w których następuje zmiana znaku wychylenia konstrukcji
        half_periods = np.diff(time_marks_of_roots)
        periods = half_periods[:-1] + half_periods[1:]
        mean_period = np.mean(periods[1:-1]) # odrzucam skrajne
        median_period = np.median(periods[1:-1])
        std_of_periods = np.std(periods[1:-1])  # standard deviation 
        return [mean_period, median_period, std_of_periods]

    def compute_envelope(self):
        self._fourier_trans()._lowpass_filter(1.5*self.main_freq)
        amplitude_envelope = np.abs(hilbert(self.Amplitude))
        self.amplitude_envelope_cutted = amplitude_envelope[int(0.02 * len(amplitude_envelope)):int(0.98 * len(amplitude_envelope))] # docinka 
        return self

    def lsq_resutls(self):
        res_lsq = least_squares(self.fun, self.x, args=(self.time_cutted, self.amplitude_envelope_cutted)) # usunięcie genrowania zmiennej lokalnej obwiedni
        return res_lsq

    def softL1_results(self):
        res_soft_l1 = least_squares(self.fun, self.x, loss='soft_l1', f_scale=0.1, args=(self.time_cutted, self.amplitude_envelope_cutted)) # jw
        return res_soft_l1

    def huber_results(self):
        res_hub = least_squares(self.fun, self.x, loss='huber', f_scale=0.1, args=(self.time_cutted, self.amplitude_envelope_cutted))  # jw
        return res_hub
    
    def _fourier_trans(self):
        """Frequency domain representation + filter cutoff computation"""
        fourierTransform = np.fft.fft(self.Amplitude_) / len(self.Amplitude_)  # Normalize amplitude
        fourierTransform = fourierTransform[range(int(len(self.Amplitude_) / 2))]  # Exclude sampling frequency
        tpCount = len(self.Amplitude_)
        values = np.arange(int(tpCount / 2))
        self.timePeriod = tpCount / self.fs  # częstotliwość próbkowania jako zmienna !!!
        frequencies = values / self.timePeriod
        self.frequecies_cutted = frequencies[int(0.001 * len(frequencies)):int(0.05 * len(frequencies))]
        self.fourierTransform_cutted = fourierTransform[int(0.001 * len(fourierTransform)):int(0.05 * len(fourierTransform))]
        loc_max_Ampl_begin_ind = np.argmax(self.fourierTransform_cutted)
        self.main_freq = self.frequecies_cutted[loc_max_Ampl_begin_ind]
        return self
 
    def _lowpass_filter(self, cutoff, order=5):
        """filtering data with lowpass filter"""
        nyq = 0.5 * self.fs
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        self.Amplitude = lfilter(b, a, self.Amplitude_)
        return self
        
    def _bandpass_filter(self, cutin, cutoff, order=5):
        """filtering data with bandpass filter"""
        nyq = 0.5 * self.fs
        normal_cutin = cutin / nyq
        normal_cutoff = cutoff / nyq
        b, a = butter(order, [normal_cutin, normal_cutoff], btype='band', analog=False)
        self.Amplitude = lfilter(b, a, self.Amplitude_)
        return self

