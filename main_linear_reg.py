from __future__ import absolute_import, division, print_function

import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import torch
matplotlib.rcParams.update({'font.size': 10})
matplotlib.rcParams.update({'lines.linewidth': 2})
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

# -------------------------------------------------------------------------------------------
#  Create data
# -------------------------------------------------------------------------------------------

# TODO: Random seed

device_id = 0
torch.cuda.device(device_id)


class TaskEnviornment():
    def __init__(self):
        self.aMean = 5
        self.aStd = 1

    def generate_task(self):
        dim = 1  # dimension of feature vector
        a = self.aStd * np.random.randn(dim, 1) + self.aMean
        return Task(a)



class DataSet():
    def __init__(self, x, y):
        self.x = torch.from_numpy(x).type(torch.float)
        self.y = torch.from_numpy(y).type(torch.float)
        self.n_samples = x.shape[0]

    def get_batch(self, batch_size):
        # Sample data batch:x_range
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


class Task():
    def __init__(self, a):
        self.xRange = (0, 10)
        self.a = a
        self.dim = a.shape[0]

    def get_samples(self, n_samples=1):
        dim = self.dim
        x = self.xRange[0] + np.random.rand(n_samples, dim) * self.xRange[1]
        a = self.a
        noise = 0.01 * np.random.randn(n_samples, dim)
        y = np.matmul(x, a) + noise
        samples = DataSet(x,y)
        return samples
    # def calc_test_error(self):



def run_task_learner(trainData, priorMu):
    taskBound = None
    postMu = None
    x = trainData.x
    y = trainData.y
    dim = x.shape[1]
    n_samples = trainData.n_samples
    matA = np.sqrt(n_samples) * torch.eye(dim) + torch.matmul(x.t(),  x)
    matB = np.sqrt(n_samples) * priorMu + torch.matmul(x.t(), y)
    postMu = torch.matmul(torch.pinverse(matA), matB)
    return postMu, taskBound



# Main Script


nPriors = 5
priorsSetMu = np.linspace(0.0, 10.0, nPriors)

taskEnv = TaskEnviornment()

# Main sequence loop
T = 1  # number of tasks
for t in range(T):
    # generate task
    task = taskEnv.generate_task()
    n_samples = 10
    trainData = task.get_samples(n_samples)
    trainData.plot()
    for i_prior in range(nPriors):
        priorMu = priorsSetMu[i_prior]
        postMu, taskBound = run_task_learner(trainData, priorMu)
