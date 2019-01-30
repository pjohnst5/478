import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots()

xVals = [-0.3, -0.2, 0.1, -0.1]
yVals = [0.2, 0.1, 0.25, 0.25]
ax.plot(xVals , yVals, 'ro')


xVals = [0.2, 0.3, 0.15, -0.25]
yVals = [0.13, 0.05, -0.05, -0.3]
ax.plot(xVals, yVals, 'g^')

ax.set_ylim(-0.4, 0.4)
ax.set_xlim(-0.4, 0.4)

x = np.arange(-0.3, 0.3, 0.01)
line = 0.9649 * x
ax.plot(x, line)

ax.set(xlabel='X axis', ylabel='Y axis',
       title='Linearly Separable, c=0.1')

ax.grid()

plt.show()
