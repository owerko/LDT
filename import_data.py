import numpy as np
import statistics
import matplotlib.pyplot as plt


def load_data(filePath):
    data = np.loadtxt(filePath, dtype=np.float64)
    t = data[:, 0]
    A = data[:, 1]
    A2 = A - statistics.mean(A)
    loc_max_A2_begin_ind = np.argmax(A2[:int(0.1 * len(t))])
    loc_max_A2_end_ind = np.argmax(A2[int(0.9 * len(t)):]) + len(A2[:int(0.9 * len(t))])
    t2 = t[loc_max_A2_begin_ind:loc_max_A2_end_ind:1]
    time = t2 - t2[0]
    Amplitude = A2[loc_max_A2_begin_ind:loc_max_A2_end_ind:1]
    return time, Amplitude


if __name__ == '__main__':
    t, A = load_data('./ztpi_data.txt')
    plt.plot(t, A)
    plt.show()
