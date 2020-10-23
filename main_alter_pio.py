import statistics
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert
from scipy.optimize import curve_fit


ztpi_data = np.loadtxt('ztpi_data.txt', dtype = np.float64)
t = ztpi_data[:,0]
A = ztpi_data[:,1]
A2 = A - statistics.mean(A)
loc_max_A2_begin_ind = np.argmax(A2[:int(0.1 * len(t))])
loc_max_A2_end_ind = np.argmax(A2[int(0.9 * len(t)):]) + len(A2[:int(0.9 * len(t))])
t2 = t[loc_max_A2_begin_ind:loc_max_A2_end_ind:1]
t3 = t2 - t2[0]
A3 = A2[loc_max_A2_begin_ind:loc_max_A2_end_ind:1]
amplitude_envelope = np.abs(hilbert(A3))


def damp_func(x, a, b, c):
    """formula for approximattion of upper hilbert envelope"""
    return a * np.exp(-b * x) + c


# Plotting the approximattion of upper hilbert envelope
param, param_conv = curve_fit(damp_func, t3, amplitude_envelope)
print(f'a: {param[0]}, b: {param[1]} c: {param[2]}')
damp_fun_points = [damp_func(i, param[0], param[1], param[2]) for i in t3]
fig, ax = plt.subplots()
ax.plot(t3, A3)
ax.plot(t3, amplitude_envelope)
ax.plot(t3, damp_fun_points)
plt.text(6, 0.12, r'a * exp(-Bt) + c', {'color': 'C2'})
plt.text(6, 0.10, f'a: {np.round(param[0], 4)},  b: {np.round(param[1], 4)},  c: {np.round(param[2], 5)}', {'color': 'C2'})
ax.set_xlabel('Time [s?]')
ax.set_ylabel('Rel. vert. displacement [mm?]')
plt.show()


# Frequency domain representation
fourierTransform = np.fft.fft(A3) / len(A3)  # Normalize amplitude 
fourierTransform = fourierTransform[range(int(len(A3) / 2))]  # Exclude sampling frequency
tpCount = len(A3)
values = np.arange(int(tpCount / 2))
timePeriod = tpCount / 200
frequencies = values / timePeriod


# Frequency domain representation - plot
fig, ax = plt.subplots()
ax.set_title('Fourier transform depicting the frequency components')
ax.plot(frequencies, abs(fourierTransform))
ax.set_xlabel('Frequency')
ax.set_ylabel('Amplitude')
plt.show()