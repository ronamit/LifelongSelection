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


class TaskEnviornment():
    def __init__(self):
        self.freqArr = [0.1, 0.2, 0.3, 0.4, 0.5]  # possible frequencies values
        nFreqs = len(self.freqArr)

        # Define the true distribution of freqs in the environment
        freqProbs = np.array([1, 1, 5, 1, 1], dtype=float)
        self.freqProbs = freqProbs / freqProbs.sum()  # normalize the probabilities
        assert nFreqs == len(self.freqProbs)

    def genrate_task(self):
        freq = np.random.choice(self.freqArr, 1, p=self.freqProbs)
        return SineTask(freq=freq)



class DataSet():
    def __init__(self, x, y):
        self.x = torch.from_numpy(x)
        self.y = torch.from_numpy(y)
        self.n_samples = x.shape[0]
    def get_batch(self, batch_size):
        # Sample data batch:
        n_samples = self.n_samples
        batch_size_curr = min(n_samples, batch_size)
        batch_inds = np.random.choice(n_samples, batch_size_curr, replace=False)
        batch_x = self.x[batch_inds]
        batch_y = self.y[batch_inds]
        return batch_x, batch_y

    def plot(self):
        #  Plots:
        fig1 = plt.figure()
        plt.plot(self.x.numpy(), self.y.numpy(), 'o')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()


class SineTask():
    def __init__(self, freq=1.0):
        self.x_range = (0, 10)
        self.amplitude = 5
        self.freq = freq

    def get_samples(self, n_samples=1000):
        x = self.x_range[0] + np.random.rand(n_samples) * self.x_range[1]
        y = self.amplitude * np.sin(x * 2 * np.pi * self.freq)
        samples = DataSet(x,y)
        return samples
    # def calc_test_error(self):



def learn_task(trainData, prior=None):
    boundVal = None
    posterior = None
    n_samples = trainData.n_samples
    # init posterior:
    freq_mu = torch.randn(1, requires_grad=True)   # the mean value of the posterior
    freq_sigma = torch.randn(1, requires_grad=True)   # the log-STD value of the posterior

    # create your optimizer
    learning_rate = 1e-1
    optimizer = torch.optim.Adam([Q_mu, Q_log_sigma], lr=learning_rate)
    nIter = 800 # number of iterations
    batch_size = 128
    # training loop:
    for i in range(nIter):
        # get batch:
        batch_x, batch_y = trainData.get_batch(batch_size)
        # Re-Parametrization:
        w


    return posterior, boundVal



taskEnv = TaskEnviornment()


nPriors = 5
prior_mu = np.linspace(0.1, 0.5, nPriors)


# Main sequence loop
T = 1  # number of tasks
for t in range(T):
    # generate task
    task = taskEnv.genrate_task()
    trainData = task.get_samples()
    trainData.plot()
