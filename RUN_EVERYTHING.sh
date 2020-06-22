#!/bin/bash

# You may need to install the following (under Debian/Ubuntu/similar):
# sudo apt-get install gfortrain liblapack (?)

# This script runs everything

# ++++++++++++++++ Compile program +++++++++++++++++++++++
gfortran -llapack compute_eig_5mom.f03

# ++++++++++++++++ Compute data (max and min wavespeeds) +++++++++++++++++

# Now computing patches of data and putting them in the same training file
#
# Note: unless we enable the ``take points randomly'', then Keras/TensorFlow
# will take points in order, and we must ensure that we pass all the data
# through the epochs and batch_size. 
# By passing first the data near equilibrium, for sure we will process it 
# at least. 

echo '> Computing some data near equilibrium ...'

q_MIN=-0.0011  # th = -3.1
q_MAX=0.0011   # th = 3.1
Nq=700

sig_MIN=0.00001
sig_MAX=0.0001
Nsig=700

./a.out $q_MIN $q_MAX $Nq $sig_MIN $sig_MAX $Nsig > outp_TRAIN.csv

# ~~~~~~~~~~~
# ~~~~~~~~~~~
# ~~~~~~~~~~~

echo '> Computing some data very close to equilibrium ...'

q_MIN=-0.00005 # theta = -0.48
q_MAX=0.00005  # theta = 0.48
Nq=500

sig_MIN=0.00001
sig_MAX=0.00005
Nsig=500

./a.out $q_MIN $q_MAX $Nq $sig_MIN $sig_MAX $Nsig >> outp_TRAIN.csv

# ~~~~~~~~~~~~
# ~~~~~~~~~~~~
# ~~~~~~~~~~~~

echo '> Computing more data in the whole range...'

q_MIN=-70.0
q_MAX=70.0
Nq=600

sig_MIN=0.00001
sig_MAX=1.0
Nsig=600

./a.out $q_MIN $q_MAX $Nq $sig_MIN $sig_MAX $Nsig >> outp_TRAIN.csv

# ~~~~~~~~~~~~
# ~~~~~~~~~~~~
# ~~~~~~~~~~~~

echo '> Computing even more data, now in the high-sigma region...'

q_MIN=-70.0
q_MAX=70.0
Nq=1000

sig_MIN=0.7
sig_MAX=1.0
Nsig=200

./a.out $q_MIN $q_MAX $Nq $sig_MIN $sig_MAX $Nsig >> outp_TRAIN.csv

# ~~~~~~~~~~~~
# ~~~~~~~~~~~~
# ~~~~~~~~~~~~

echo '> Computing some last data, very close to equilibrium ...'


q_MIN=-0.00008 # theta = -0.732
q_MAX=0.00008  # theta = 0.732
Nq=500

sig_MIN=0.00001
sig_MAX=0.00004
Nsig=500

./a.out $q_MIN $q_MAX $Nq $sig_MIN $sig_MAX $Nsig >> outp_TRAIN.csv



# ++++++++++++++++++++ Train neural net +++++++++++++++++++
echo '> Training neural net...'
python3 train_net.py

# ++++++++++++++++++++ Check the prediction ++++++++++++++++++++
echo '> Checking the predictions (using less data) in the whole large domain...'

q_MIN=-70.0
q_MAX=70.0
Nq=100

sig_MIN=0.00001
sig_MAX=1.0
Nsig=100

./a.out $q_MIN $q_MAX $Nq $sig_MIN $sig_MAX $Nsig > outp_TEST.csv

octave octaLoadAndUseNet.m

