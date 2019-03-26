from __future__ import absolute_import, division, print_function

import numpy as np
import matplotlib
from matplotlib import pyplot as plt
matplotlib.rcParams.update({'font.size': 10})
matplotlib.rcParams.update({'lines.linewidth': 2})
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


# -------------------------------------------------------------------------------------------
#  Create data
# -------------------------------------------------------------------------------------------

# TODO: Random seed


# generate inputs
m = 1000  # data-set size


class SineTask():
    def __init__(self, n_samples=1000, freq=1.0):
        self.x_range = (0, 10)
        self.amplitude = 5
        self.n_samples = n_samples
        self.freq = freq
        self.x = self.x_range[0] + np.random.rand(m) * self.x_range[1]
        self.y = np.empty_like(self.x)
        for i_point in range(m):
            self.y[i_point] = self.amplitude * np.sin(self.x[i_point] * 2 * np.pi * self.freq)

    def plot(self):
        #  Plots:
        fig1 = plt.figure()
        plt.plot(self.x, self.y, 'o')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()

    # def calc_test_error(self):



freq_arr = np.linspace(0.1, 0.5, 5)  # possible frequencies
n_freqs = len(freq_arr)
i_freq = np.random.randint(0, len(freq_arr) - 1)
freq = freq_arr[i_freq]

taskData = SineTask(n_samples=1000, freq=freq)
taskData.plot()


# Main sequence loop
T = 5  # number of tasks
# for t in range(T):
#