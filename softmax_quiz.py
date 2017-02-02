"""Softmax."""

import numpy as np

scores1 = [3.0, 1.0, 0.2]
scores2= [1.0, 2.0, 3.0]
scores3 = np.array([[1, 2, 3, 6],
                   [2, 4, 5, 6],
                   [3, 8, 7, 6]])

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    D=np.sum(np.exp(x))
    P=(np.exp(x)/D)
    #pass  # TODO: Compute and return softmax(x)
    return P

print(softmax(scores1))
print(softmax(scores2))
print(softmax(scores3))

# Plot softmax curves
import matplotlib.pyplot as plt
x = np.arange(-2.0, 6.0, 0.1)
scores = np.vstack([x, np.ones_like(x), 0.2 * np.ones_like(x)])

plt.plot(x, softmax(scores).T, linewidth=2)
plt.show()
