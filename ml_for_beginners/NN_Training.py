#reference the measurements in measurements.png and trainingModel.png

#lets train a network to predict someone's gender given their weight and height

# we will rep Male with 0 and Female with 1, we will also shift the data to make it easier to use

#Loss

#before we train our network, we need a way to quantify how "good" it's doing so that it can try to do "better", thats what the loss is

#we will use mean squared error loss

#see MSE.png

#An Example Loss Calculation

#lets say our network always outputs 00, its confident all humans are Male. What would our loss be?


import numpy as np


def mse_loss(y_true, y_pred):
    # y_true and y_pred are numpy arrays of the same length.
    return ((y_true-y_pred)**2).mean()

y_true = np.array([1,0,0,1])
y_pred = np.array([0,0,0,0])

print(mse_loss(y_true,y_pred)) # 0.5


# we now have a clear goal: minimize the loss of the neural network. We know we can change the network's weights and biases to influence its prediction, but how do
# we change the network's weights and biases to influence its predictions, but how do we do so in a way that decreases loss