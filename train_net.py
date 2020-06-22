#
# This script trains a neural net to reproduce the maximum and minimum wavespeeds of the maximum-entropy 
# moments problem. Data is stored in the outp_TRAIN.csv file. The format is:
# THETA=asinh(10000*q_star)   |   XI=log10(sig)  |  LOG10(MAXVAL(eigProcessed))    |    LOG10(MAXVAL(-eigProcessed))
#
# Note: you can use either tensorflow or theano. This script uses Keras.
# Set backend for Keras in ~/.keras, to "tensorflow" or "theano"
# For theano, also install pyyaml.


import numpy as np
from numpy import loadtxt

from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers
from keras import metrics
from keras import losses

import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

# -------------------------------------------

def scale_shift_01(vec):
   # This function scales the elements of the input array "vec" to a "0 to 1" range.
   # Also provides minimum and maximum, that we need for unscaling.
   vec_min      = np.min(vec)
   vec_max      = np.max(vec)
   vec_rescaled = (vec - vec_min)/(vec_max - vec_min)
   return vec_rescaled, vec_min, vec_max

# ---

def unscale_unshift_01(vec_scaled, vec_min, vec_max):
   # This function un-scales from the range "0 to 1" back to the original range.
   # Needs minimum and maximum values of the original.
   return vec_min + vec_scaled*(vec_max - vec_min)

# ---

def scale_shift_1m1(vec):
   # This function scales the elements of the input array "vec" to a "-1 to 1" range.
   # Also provides minimum and maximum, that we need for unscaling.
   vec_min      = np.min(vec)
   vec_max      = np.max(vec)
   vec_rescaled = 2.0*(vec - vec_min)/(vec_max - vec_min) - 1.0
   return vec_rescaled, vec_min, vec_max

# ---

def unscale_unshift_1m1(vec_scaled, vec_min, vec_max):
   # This function un-scales from the range "-1 to 1" back to the original range.
   # Needs minimum and maximum values of the original.
   return vec_min + (vec_scaled + 1.0)*(vec_max - vec_min)/2.0

# -------------------------------------------

# Load dataset
dataset = loadtxt('outp_TRAIN.csv', delimiter=',')

X  = dataset[:,0] # theta = asinh(10000 q_star)
Y  = dataset[:,1] # xi = log10(sigma)
Z1 = dataset[:,2] # Max wavespeed: log10(max(w))
Z2 = dataset[:,3] # Min wavespeed: - log10(-min(w))

Ntrain = np.size(X)

# Rescaling data
X_scal_train, X_min, X_max = scale_shift_1m1(X)
Y_scal_train, Y_min, Y_max = scale_shift_1m1(Y)
Z1_scal_train, Z1_min, Z1_max = scale_shift_1m1(Z1)
Z2_scal_train, Z2_min, Z2_max = scale_shift_1m1(Z2)

XY_train = np.zeros((Ntrain,2))
ZZ_train = np.zeros((Ntrain,2))

XY_train[:,0] = X_scal_train 
XY_train[:,1] = Y_scal_train

ZZ_train[:,0] = Z1_scal_train 
ZZ_train[:,1] = Z2_scal_train

# Create neural net model with Keras
model = Sequential()
model.add(Dense(5, input_dim=2, activation = 'tanh'))
model.add(Dense(10, activation = 'tanh'))
model.add(Dense(5, activation='tanh'))
model.add(Dense(2, activation='linear'))

# Compile the Keras model
#### model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.compile(optimizers.Adam(lr=0.005), loss=losses.mean_squared_error, metrics=['mse'])
#### model.compile(optimizers.SGD(lr=0.01, momentum=0.3), loss=losses.mean_absolute_error, metrics=['mse'])
#### model.compile(optimizers.SGD(lr=0.01, momentum=0.3), loss=losses.mean_squared_logarithmic_error, metrics=['mse'])

# Fit the Keras model on dataset
history = model.fit(XY_train, ZZ_train, epochs=2000, batch_size=10000, verbose=2)

# Save model
model.save('saved_keras_model.h5')

# Evaluate the model (to have a PRELIMINARY idea of the quality)
_, accuracy = model.evaluate(XY_train,ZZ_train)
print('Accuracy: %.5f %%' %(accuracy*100))

# # PLOT FITTING HISTORY
# plt.title('Loss / Mean Squared Error')
# plt.plot(history.history['loss'], label='train')
# plt.plot(history.history['val_loss'], label='test')
# plt.legend()
# plt.show()
# 
