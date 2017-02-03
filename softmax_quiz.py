"""Softmax."""
import numpy as np
import matplotlib.pyplot as plt

scores1 = [3.0, 1.0, 0.2]
scores2 = [1.0, 2.0, 3.0]
scores3 = np.array([ [1, 2, 3, 6], [2, 4, 5, 6], [3, 8, 7, 6] ])

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    D=np.sum(np.exp(x), axis=0)
    P=(np.exp(x)/D)
    return P

print(softmax(scores1))
print(softmax(scores2))
print(softmax(scores3))

# Plot softmax curves
x = np.arange(-2.0, 6.0, 0.1)
scores = np.vstack([x, np.ones_like(x), 0.2*np.ones_like(x)])
print("Stacked scores 1st row: {}".format(scores[:,0]))
# Each column in scores matrix is a sample of 3 elements

# Next figure shows how probabilty values after applying softmax function to scores changes depending on the relative importance of the scores in each sample of 3 elements
plt.figure
plt.plot(x, softmax(scores).T, linewidth=2)
plt.legend(['x', 'ones_like(x)', '0.2*ones_like(x)'], loc=0)
plt.show()

# If you increase the size of your output scores, your classifier becomes very confident about its softmax-probability predictions ==> probabilities density tends to a binary distribution
scores2 = np.vstack([np.ones_like(x)*10, x*10])
plt.figure
plt.plot(x*10, softmax(scores2).T, linewidth=2)
plt.legend(['ones_like(x)*10', 'x*10'], loc=0)
plt.show()

# But if you reduce the size of your output scores, your classifier becomes very insecure about its softmax-probability predictions ==> probabilities density tends to a uniform distribution
scores3 = np.vstack([np.ones_like(x)/10, x/10])
plt.figure
plt.plot(x/10, softmax(scores3).T, linewidth=2)
plt.legend(['ones_like(x)/10', 'x/10'], loc=0)
plt.show()