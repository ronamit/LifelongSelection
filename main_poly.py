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

p = 3  # Polynomial degree
a_range = (-10, 10) # range of Polynomial coefficients
a = a_range[0] + a_range[1] * np.random.rand(p) # Polynomial coefficients

# generate inputs
m = 60  # data-set size
x_range = (0, 10)
x = x_range[0] + np.random.rand(m) * x_range[1]
y = np.empty_like(x)
for i in range(m):
    y[i] = np.sum([x[i]**k * a[k] for k in range(p)])


#  Plots:
fig1 = plt.figure()
plt.plot(x, y,  'o')
plt.xlabel('x')
plt.ylabel('y')

plt.show()
