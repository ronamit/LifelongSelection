from __future__ import absolute_import, division, print_function

import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import torch
matplotlib.rcParams.update({'font.size': 10})
matplotlib.rcParams.update({'lines.linewidth': 2})
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


# TODO: Random seed

device_id = 0
torch.cuda.device(device_id)

# -------------------------------------------------------------------------------------------
#  Task-environment class
# -------------------------------------------------------------------------------------------
class TaskEnvironment():
    def __init__(self):
        self.aMean = 4
        self.aStd = 0.1
        self.noiseStd = 5.0
        self.xRange = (0, 10)

    def generate_task(self):
        dim = 1  # dimension of feature vector
        a = self.aStd * np.random.randn(dim, 1) + self.aMean
        return Task(self, a)

# -------------------------------------------------------------------------------------------
#  Task class
# -------------------------------------------------------------------------------------------
class Task():
    def __init__(self, taskEnv, a):
        self.xRange = (0, 10)
        self.a = a
        self.dim = a.shape[0]
        self.noiseStd = taskEnv.noiseStd
        self.xRange = taskEnv.xRange


    def get_samples(self, n_samples=1):
        dim = self.dim
        x = self.xRange[0] + np.random.rand(n_samples, dim) * self.xRange[1]
        a = self.a
        noise = self.noiseStd * np.random.randn(n_samples, dim)
        y = np.matmul(x, a) + noise
        samples = DataSet(x,y)
        return samples

    def est_test_error(self, postMu, postVar):
        n_samples_test = 1000
        self.get_samples(n_samples_test)
        x = trainData.x
        y = trainData.y
        test_err = (1/n_samples_test) * (torch.norm(y - torch.matmul(x, postMu))**2
                                         + postVar * torch.norm(x)**2)
        return test_err




# -------------------------------------------------------------------------------------------
#  DataSet class
# -------------------------------------------------------------------------------------------
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


# -------------------------------------------------------------------------------------------
#  Task-learning function
# -------------------------------------------------------------------------------------------
def run_task_learner(trainData, priorMu, priorVar):
    # note: we assume priorVar == postVar
    taskBound = None
    postMu = None
    x = trainData.x
    y = trainData.y
    dim = x.shape[1]
    n_samples = trainData.n_samples
    regFactor = np.sqrt(n_samples) / (2 * priorVar)
    matA = regFactor * torch.eye(dim) + torch.matmul(x.t(),  x)
    matB = regFactor * priorMu + torch.matmul(x.t(), y)
    # postMu = torch.from_numpy(np.linalg.solve(matA, matB))
    postMu = torch.matmul(torch.pinverse(matA), matB)
    taskBound = (1/n_samples) * (torch.norm(matB - matA.t() * postMu)**2 + regFactor * torch.norm(postMu - priorMu)**2)
    # TODO: calculate exact bound for comparison with actual results
    return postMu, taskBound

def run_no_prior_learner(trainData):
    # note: we assume priorVar == postVar
    x = trainData.x
    y = trainData.y
    dim = x.shape[1]
    n_samples = trainData.n_samples
    matA =  torch.matmul(x.t(),  x)
    matB =  torch.matmul(x.t(), y)
    est = torch.matmul(torch.pinverse(matA), matB)
    return est
# -------------------------------------------------------------------------------------------
#  # Main Script
# -------------------------------------------------------------------------------------------
# nPriors = 5
# priorsSetMu = np.linspace(0.0, 10.0, nPriors)
priorsSetMu = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
nPriors = priorsSetMu.shape[0]
priorVar = 0.2**2  # TODO: make prior variance different so it will help the results which the prior is correct
postVar = priorVar

cumulativeBound = np.zeros(nPriors)

taskEnv = TaskEnvironment()

n_samples = 2 # number of samples per task TODO: draw at random for each task
# -------------------------------------------------------------------------------------------
#   Main lifelong learning loop
# -------------------------------------------------------------------------------------------
T = 100  # number of tasks
for t in range(T):
    # generate task
    task = taskEnv.generate_task()
    # print(task.a)
    trainData = task.get_samples(n_samples)
    # trainData.plot()
    for i_prior in range(nPriors):
        priorMu = priorsSetMu[i_prior]
        postMu, taskBound = run_task_learner(trainData, priorMu, priorVar)
        cumulativeBound[i_prior] += taskBound


# -------------------------------------------------------------------------------------------
# hyper-posterior calculation
# -------------------------------------------------------------------------------------------

print([(priorsSetMu[k], cumulativeBound[k] / T) for k in range(nPriors)])

hyperPrior = np.ones(nPriors) / nPriors
alpha = 1 / np.sqrt(T) + 1 / n_samples  # assuming all tasks have the same number of samples
hyperPosterior = (hyperPrior ** alpha) * np.exp(-(1/T) * cumulativeBound)
hyperPosterior = hyperPosterior / hyperPosterior.sum()
print(hyperPosterior)

transferBound = np.sum(hyperPosterior * (cumulativeBound / T + alpha * np.log(hyperPosterior / hyperPrior)))
print('Transfer Bound: {}'.format(transferBound))

# -------------------------------------------------------------------------------------------
# meta-testing
# -------------------------------------------------------------------------------------------
nReps = 100
errVecAlg = np.zeros(nReps)
errVec0prior = np.zeros(nReps)
errVecNoPrior = np.zeros(nReps)
errVecAlgPeak = np.zeros(nReps)

for iRep in range(nReps):
    # draw prior from hyper-posterior
    priorMu = np.random.choice(priorsSetMu, 1, p=hyperPosterior)[0]
    # generate task
    task = taskEnv.generate_task()
    trainData = task.get_samples(n_samples)
    # Check expected error when using learned hyper-posterior
    postMu, taskBound = run_task_learner(trainData, priorMu, priorVar)
    errVecAlg[iRep] = task.est_test_error(postMu, postVar)

    # Check expected error when using learned hyper-posterior
    postMu, taskBound = run_task_learner(trainData, priorMu, priorVar)
    errVecAlgPeak[iRep] = task.est_test_error(postMu, 0)

    # Check expected error when using prior #0
    postMu, taskBound = run_task_learner(trainData, priorsSetMu[0], priorVar)
    errVec0prior[iRep] = task.est_test_error(postMu, postVar)

    # Check expected error when using no prior
    est_a = run_no_prior_learner(trainData)
    errVecNoPrior[iRep] = task.est_test_error(est_a, 0)


print('Estimated transfer-error, using Lifelong Alg: {}'.format(errVecAlg.mean()))
print('Estimated transfer-error, using Lifelong Alg +peak-posterior: {}'.format(errVecAlgPeak.mean()))
print('Estimated transfer-error, using prior #0: {}'.format(errVec0prior.mean()))
print('Estimated transfer-error, not using prior: {}'.format(errVecNoPrior.mean()))





