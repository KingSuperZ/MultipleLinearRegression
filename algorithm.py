""" Multiple Linear Regression

We assume that there are only two inputs/features (x1,x2).

In this algorithm we take data which contains an X (features) and y (targets), 
but the X has two features which we call x1 and x2. 

The algorithm we assume that the best equation for this algorithm is: 
out = w0 + w1x1 + w2x2. The unknown values are w0, w1 and w2.

Before we try out the algorithm on finding unknowns, we need to train the data
by using given values of x1 and x2 to see if the variable out is close to its
respective target.

Once the training is complete, we test new values for x1 and x2 to predict what
the variable out would yield.
"""

import numpy as np
import matplotlib.pyplot as plt

# Datapoints
X = np.array([[1,2],[2,1],[3,6],[4,3],[5,-1],[6,0]]) # Features with 6 rows and 2 columns
y = np.array([[9],[8],[25],[18],[8],[13]])# Targets with 1 row and six columns

# For the algorithm Multiple Linear Regression (MLR) we used the equation 
# out = w0 + w1x1 + w2x2 (equation of a plane). The reason for there being only 
# this many variables is because it best fits the number of features we have in 
# our data. The unknowns are the parameters w0, w1, w2. We know x1 and x2 as the 
# features given in the data with x1 being one column and x2 being another. We 
# randomly pick the values for w0, w1, and w2 and use Gradient Descent to 
# slowly improve the values so that the variable out is as close to y as possible

# Guessed Parameters
w0 = 2 
w1 = 4
w2 = 6

# The learning rate is the rate at which the parameters reach their desired result
# and a low learning rate means that the parameters slowly change and a high learning
# rate means that the parameters change more drastically.
lr = 0.01

# Number of iterations in which the parameters go through. A singular iteration
# means that the parameters go through each row once.
n_epochs = 4000 

# This part of MLR is where the parameters are being trained/learned to reach values that
# make it so that the variable out is as close to y as possible.
for epoch in range(n_epochs):
    for i in range(len(y)):
        out = w0 + w1*X[i,0] + w2*X[i,1]
        error = out - y[i]
        w0 = w0 - lr*(error)
        w1 = w1 - lr*(error)*X[i,0]
        w2 = w2 - lr*(error)*X[i,1]
print(w0)
print(w1)
print(w2)

# We reshaped the features so that the dimensions can be used for the 3D plot
x1 = X[:,0]
x2 = X[:,1]
X1,X2 = np.meshgrid(x1,x2)

# The equation represents the plane of best fit
out = w0 + w1*X1 + w2*X2

# Here the plane of best is being shown as a wireframe plot
fig, ax = plt.subplots(subplot_kw={"projection":"3d"})
ax.plot_wireframe(X1,X2,out,cmap = "Blues", alpha = 0.5, label = "Plane of Best Fit")

# Here is datapoints are being plotted with the x and y axis contained the 
# features and the z axis contained the targets.
ax.scatter(x1,x2,y, c = "red", label = "Datapoints")

ax.set_xlabel("X1")
ax.set_ylabel("X2")
ax.set_zlabel("Out")
ax.view_init(elev = 10, azim = 45)
ax.legend()
