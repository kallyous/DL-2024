# Package imports
import numpy as np
import matplotlib.pyplot as plt
from testCases_v2 import *
import sklearn
import sklearn.datasets
import sklearn.linear_model
from planar_utils import plot_decision_boundary, sigmoid, \
                         load_planar_dataset, load_extra_datasets



np.random.seed(1)  # set a seed so that the results are consistent

X, Y = load_planar_dataset()

# Visualize the data:
plt.scatter(X[0, :], X[1, :])  # c=Y, s=40, cmap=plt.cm.Spectral);
plt.xlabel('x')
plt.ylabel('y')

### START CODE HERE ### (≈ 3 lines of code)
shape_X = None
shape_Y = None
m = None  # training set size
### END CODE HERE ###

print(f'The shape of X is: {shape_X}')
print(f'The shape of Y is: {shape_Y}')
print(f'I have m = {m} training examples!')
