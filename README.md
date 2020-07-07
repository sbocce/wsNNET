# wsNNET

# Some explanation on how stuff works.
############################################

############################################
# PREFACE:
############################################

The file 5mom.mac computes the analytical Jacobian with symbolic computation, by using Maxima. 
The Jacobian is then implemented into the compute_eig_5mom.f90, that uses LAPACK to compute its eigenvalues.
The directory "SAVED_NNET_MODELS" stores models trained to fit the 5 moments system, in the range:
q_star in (-70,70)
and sigma in (1e-5, 1).

The TOOLS directory contains some Octave tools such as a script to convert a neural net keras model into 
Fortran matrices.

The rest of this file describes how to run the whole process training of a neural network: generation of data, 
training and verification.

############################################
# RUNNING THE WHOLE PROCESS
############################################

If you want to run the whole process, just run:

$ ./RUN_EVERYTHING.sh

Here is what it will do.

----------------------------------
STEP 1: GENERATING THE EIGENVALUES

compute_eig_5mom.f90 

  Computes the eigenvalues from the analytical Jacobian (implemented in the source code).
  Compile with: 

  $ gfortran compute_eig_5mom.f90 -llapack

  The output of this program will be in CSV format, and you can save it as

  $ ./a.out qMIN qMAX Nq sigMIN sigMAX Nsig > outp.csv

  Where qMIN, qMAX, sigMIN, sigMAX are the minimum and maximum values for the dimensionless
  heat flux and for the sigma parameter, and Nq and Nsig are the number of points.
  The output variables are scaled as to make the wavespeeds (eigenvalues of the Jacobian)
  a simple and smooth function, that the neural net can actually fit well.
  See the end of the compute_eig_5mom.f90 file and you'll see what is actually printed.

----------------------------------
STEP 2: TRAINING THE NEURAL NET

train_net.py

  Then, train the net and export the model.
  Run with:

  $ python3 train_net.py

----------------------------------
STEP 3: CHECK QUALITY OF THE TRAINED NEURAL NET

octaLoadAndUseNet.m

  Loads a net model saved with Keras and uses it by calling the function apply_saved_net.m
  Note that you need to specify an array of activation functions (which for some reason 
  I couldn't export from Keras).
  The length should match the number of layers in the neural net.
  

apply_saved_net.m
  
  Extracts the matrices from the saved Keras model and applies it to the requested data.

 
