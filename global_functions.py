import numpy as np


def Branin(x):
    first_term = (x[1] - 5.1 / (4 * np.pi ** 2) * x[0] ** 2
                  + 5.0 / np.pi * x[0] - 6) ** 2
    second_term = 10 * (1.0 - 1.0 / (8 * np.pi)) * np.cos(x[0])
    return first_term + second_term + 10.0


def Modified_Branin(x):
    return Branin(x) + 20 * x[0] - 30 * x[1]


def Bowl(x):
    x = np.array(x)
    center = np.array([-3, -3])
    R = 10
    return 0.5 * (np.linalg.norm(x-center) ** 2 - R ** 2)


def Inverted_Bowl(x):
    return - Bowl(x)


def SinQ(x):
    return np.sin((x[0] ** 2 + x[1] ** 2)/10.0)
