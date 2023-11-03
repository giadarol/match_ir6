import numpy as np

def _sigmoid(x):

    if x < 0:
        return 0
    if x < 1.:
        return 2 /np.pi - 2 /np.pi * np.cos(np.pi * x / 2)
    else:
        return x + 2 / np.pi - 1

def _sigmoid(x):
        if x < 0:
            return 0
        if x < 1.:
            #return 3 * x**3 - 2 * x**4
            #return 6 * x**3 - 8 * x**4 + 3 * x**5
            return 10 * x**3 - 20 * x**4 + 15 * x**5 - 4 * x**6
        else:
            return x

def _poly(x):
     return 3 * x**3 - 2 * x**4

def _sigmoid(x):
        x_cut = 1/16 + np.sqrt(33)/16
        if x < 0:
            return 0
        if x < x_cut:
            return _poly(x)
        else:
            return x - x_cut + _poly(x_cut)

x = np.linspace(-1, 2, 100000)
y = np.array([_sigmoid(xx) for xx in x])

import matplotlib.pyplot as plt
plt.close('all')
ax1 = plt.subplot(2, 1, 1)
ax1.plot(x, y)
ax1.grid(True)
ax2 = plt.subplot(2, 1, 2, sharex=ax1)
ax2.plot(x, np.gradient(y, x))

plt.show()