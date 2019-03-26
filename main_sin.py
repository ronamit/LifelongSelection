from __future__ import absolute_import, division, print_function

import numpy as np
import matplotlib
from matplotlib import pyplot as plt
matplotlib.rcParams.update({'font.size': 10})
matplotlib.rcParams.update({'lines.linewidth': 2})
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

import torch

# -------------------------------------------------------------------------------------------
#  Create data
# -------------------------------------------------------------------------------------------

# TODO: Random seed

device_id = 0
torch.cuda.device(device_id)

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
        self.y = self.amplitude * np.sin(self.x * 2 * np.pi * self.freq)

    def plot(self):
        #  Plots:
        fig1 = plt.figure()
        plt.plot(self.x, self.y, 'o')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()

    def learn(self, prior=None):
        boundVal = None
        posterior = None

        Q_mu = torch.randn(1, requires_grad=True)
        return posterior, boundVal


    # def calc_test_error(self):



freqArr = [0.1, 0.2, 0.3, 0.4, 0.5]  # possible frequencies values
nFreqs = len(freqArr)

# Define the true distribution of freqs in the environment
freqDist = np.array([1,1,5,1,1], dtype=float)
freqDist = freqDist / freqDist.sum
assert nFreqs == len(freqDist)

i_freq = np.random.randint(0, len(freqArr) - 1)
freq = freqArr[i_freq]

taskData = SineTask(n_samples=1000, freq=freq)
taskData.plot()

nPriors = 5
prior_mu = np.linspace(0.1, 0.5, nPriors)


# Main sequence loop
T = 5  # number of tasks
# for t in range(T):
#