
# coding: utf-8

# # Simple Linear Regression. Minimal example

# ### Import the relevant libraries

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# ### Generate random input data to train on

# In[ ]:


# Declare a variable containing the size of the training set we want to generate.
observations = 1000

# Creating two variables as inputs(xs and zs). 
xs = np.random.uniform(-10, 10, (observations,1))
zs = np.random.uniform(-10, 10, (observations,1))

# Combining both the variables to generate a single matrix
inputs = np.column_stack((xs,zs))

print (inputs.shape)


# ### Generate the targets we will aim at

# In[ ]:


# Adding a small random noise to the function.
noise = np.random.uniform(-1, 1, (observations,1))

# Produce the targets according to the f(x,z) = 2x - 3z + 5 + noise definition.
targets = 2*xs - 3*zs + 5 + noise

print (targets.shape)


# ### Plot the training data

# In[ ]:


# In order to use the 3D plot, the objects should have a certain shape, so we reshape the targets.
# The proper method to use is reshape and takes as arguments the dimensions in which we want to fit the object.
targets = targets.reshape(observations,)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.plot(xs, zs, targets)

# Set labels
ax.set_xlabel('xs')
ax.set_ylabel('zs')
ax.set_zlabel('Targets')

# You can fiddle with the azim parameter to plot the data from different angles.
ax.view_init(azim=100)

plt.show()

# We reshape the targets back to the shape that they were in before plotting.
targets = targets.reshape(observations,1)


# ### Initialize variables

# In[ ]:


# We will initialize the weights and biases randomly in some small initial range.
init_range = 0.1

weights = np.random.uniform(low=-init_range, high=init_range, size=(2, 1))
biases = np.random.uniform(low=-init_range, high=init_range, size=1)

print (weights)
print (biases)


# ### Set a learning rate

# In[ ]:


# Setting some small learning rate
learning_rate = 0.02


# ### Train the model

# In[ ]:


for i in range (100):
    
    # As per the linear model: y = xw + b equation
    outputs = np.dot(inputs,weights) + biases
    
    # The deltas are the differences between the outputs and the targets
    deltas = outputs - targets
        
    # Considering the L2-norm loss. Rescaling it by dividing it by 2 and further by the number of observations
    loss = np.sum(deltas ** 2) / 2 / observations
    
    print (loss)
    
    # Rescaling the deltas
    deltas_scaled = deltas / observations
    
    # Apply the gradient descent update rule
    # The weights are 2x1, learning rate is 1x1 (scalar), inputs are 1000x2, and deltas_scaled are 1000x1
    # We must transpose the inputs so that we get an allowed operation.
    weights = weights - learning_rate * np.dot(inputs.T,deltas_scaled)
    biases = biases - learning_rate * np.sum(deltas_scaled)


# ### Print weights and biases and see if we have worked correctly.

# In[ ]:


print (weights, biases)


# In[ ]:


plt.plot(outputs,targets)
plt.xlabel('outputs')
plt.ylabel('targets')
plt.show()

