import numpy as np
from scipy.optimize import least_squares
from matplotlib import pyplot as plt

def gen_data(t, A, d, s, noise=0, n_outliers=0, random_state=0):
    y = A*np.exp(d*-t)*np.cos(s*np.pi*t)

    rnd = np.random.RandomState(random_state)
    error = noise * rnd.randn(t.size)
    outliers = rnd.randint(0, t.size, n_outliers)
    error[outliers] *= 10
    return y + error

A = 10
d = 1
s = 2.0

t_min = 0
t_max = 10
n_points = 50

t_train = np.linspace(t_min, t_max, n_points)
y_train = gen_data(t_train, A, d, s, noise=0.1, n_outliers=3)

def fun(x, t, y):
    return x[0] * np.exp(x[2] * t) * np.cos(x[1] * np.pi * t) - y

x0 = np.array([1.0, 1.0, 0.0])

res_lsq = least_squares(fun, x0, args=(t_train, y_train))

res_soft_l1 = least_squares(fun, x0, loss='soft_l1', f_scale=0.1,
                            args=(t_train, y_train))
res_log = least_squares(fun, x0, loss='cauchy', f_scale=0.1,
                        args=(t_train, y_train))
res_huber = least_squares(fun, x0, loss='huber', f_scale=0.1,
                        args=(t_train, y_train))
res_huber = least_squares(fun, x0, loss='arctan', f_scale=0.1,
                        args=(t_train, y_train))

t_test = np.linspace(t_min, t_max, n_points * 10)
y_true = gen_data(t_test, A, d, s)
y_lsq = gen_data(t_test, *res_lsq.x)
y_soft_l1 = gen_data(t_test, *res_soft_l1.x)
y_log = gen_data(t_test, *res_log.x)

plt.plot(t_train, y_train, 'p')
plt.plot(t_test, y_true, 'r', linewidth=1, label='true')
plt.plot(t_test, y_lsq, label='linear loss')
#plt.plot(t_test, y_soft_l1, label='soft_l1 loss')
#plt.plot(t_test, y_log, label='huber')
plt.plot(t_test, y_log, label='cauchy loss')
#plt.plot(t_test, y_log, label='arctan')
plt.xlabel("t")
plt.ylabel("y")
plt.legend()
plt.show()