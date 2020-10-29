import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert
from scipy.optimize import least_squares
from import_data import load_data


def lsq_resutls(t, A):
    amplitude_envelope = np.abs(hilbert(A))
    x0 = np.array([1.0, 1.0, 1.0])
    res_lsq = least_squares(fun, x0, args=(t, amplitude_envelope))
    return res_lsq


def softL1_results(t, A):
    amplitude_envelope = np.abs(hilbert(A))
    x0 = np.array([1.0, 1.0, 1.0])
    res_soft_l1 = least_squares(fun, x0, loss='soft_l1', f_scale=0.1, args=(t, amplitude_envelope))
    return res_soft_l1


def huber_results(t, A):
    amplitude_envelope = np.abs(hilbert(A))
    x0 = np.array([1.0, 1.0, 1.0])
    res_hub = least_squares(fun, x0, loss='huber', f_scale=0.1, args=(t, amplitude_envelope))
    return res_hub



def fun(x, t, y):
    return x[0] * np.exp(-x[1] * t) + x[2] - y


def gen_data(t, a, b, c):
    return a * np.exp(-b * t) + c


if __name__ == '__main__':
    t, A = load_data('./ztpi_data.txt')
    y_lsq = gen_data(t, *lsq_resutls(t, A).x)
    y_soft_l1 = gen_data(t, *softL1_results(t, A).x)
    y_huber = gen_data(t, *huber_results(t, A).x)

    print('To samo co u Piotra inna biblioteka - zwykla matoda najmniejszych kawdratow - wynik ten sam')
    print(f'LSF a: {lsq_resutls(t, A).x[0]:.3}, b: {lsq_resutls(t, A).x[1]:.3} c: {lsq_resutls(t, A).x[2]:.3}')
    print('Przyklad funkcji odpornej 1')
    print(
        f'Soft L1 a: {softL1_results(t, A).x[0]:.3}, b: {softL1_results(t, A).x[1]:.3} c: {softL1_results(t, A).x[2]:.3}')
    print('Przyklad funkcji odpornej 2')
    print(f'Huber a: {huber_results(t, A).x[0]:.3}, b: {huber_results(t, A).x[1]:.3} c: {huber_results(t, A).x[2]:.3}')

    plt.plot(t, A)
    plt.plot(t, y_lsq, '--', label='linear loss')
    plt.plot(t, y_soft_l1, '-.', label='soft_l1 loss')
    plt.plot(t, y_huber, label='huber loss')
    plt.xlabel("t")
    plt.ylabel("y")
    plt.legend()
    plt.show()

    t, A = load_data('./murckowskaW.txt')
    plt.plot(t, A)
    plt.show()
