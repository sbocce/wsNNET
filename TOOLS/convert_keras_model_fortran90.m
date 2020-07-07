close all
clear
clc

%
% This script loads a Keras model (take for example one of the models saved in the "SAVED_NNET_MODELS") and
% writes the matrices into fortran90 syntax. 
% So, you just have to copy-paste them in your code.
% NOTE THAT ACTIVATION FUNCTIONS ARE TO BE IMPLEMENTED SEPARATELY! You need to do that yourself.

% Load saved keras model
model = load('saved_keras_model.h5');

% Open file to write the model to  
fid = fopen('exported_model.f90', 'w');
  
% ############ Loop through the layers of the neural net ############

fprintf(fid, 'ATTENTION, MATRICES ARE WRITTEN IN SINGLE PRECISION! SEE THE "e" SYMBOL? CONVERT MANUALLY WITH SUBSTITUTE ALL!\n')
fprintf(fid, 'ATTENTION, MATRICES ARE WRITTEN IN SINGLE PRECISION! SEE THE "e" SYMBOL? CONVERT MANUALLY WITH SUBSTITUTE ALL!\n')
fprintf(fid, 'ATTENTION, MATRICES ARE WRITTEN IN SINGLE PRECISION! SEE THE "e" SYMBOL? CONVERT MANUALLY WITH SUBSTITUTE ALL!\n\n\n')

% ====== First round: write variables declaration

counter = 0;

for [val, key] = model.model_weights

  % +++++++++++ Layer number +++++++++++++

  counter = counter + 1;

  % ++++++++++ Extract weights matrix and biases ++++++++++++++

  s = getfield(val, key); % extract the two fields at this layer
  
  W = s.kernel_0;
  b = s.bias_0';

  Nb = numel(b);
  N_W2 = size(W,2);

  fprintf(fid, 'REAL(KIND=8), DIMENSION(%d) :: b%d\n', Nb, counter)
  fprintf(fid, 'REAL(KIND=8), DIMENSION(%d,%d) :: W%d\n', Nb, N_W2, counter)
  fprintf(fid, '\n')

end

% ===== Second round: now write matrices ======

fprintf(fid, '\n')
fprintf(fid, '! ------- \n')
fprintf(fid, '\n')


counter = 0;

for [val, key] = model.model_weights

  % +++++++++++ Layer number +++++++++++++

  counter = counter + 1;

  % ++++++++++ Extract weights matrix and biases ++++++++++++++

  s = getfield(val, key); % extract the two fields at this layer
  
  W = s.kernel_0;
  b = s.bias_0';

  % ++++++++++ Save everything to file

  Nb = numel(b);
  N_W2 = size(W,2);

  fprintf(fid, '\n')

  % Write b
  for i = 1:Nb
    fprintf(fid, 'b%d(%d) = %1.14e\n',counter, i, b(i))
  end

  fprintf('\n\n')
  % Write W
  for i = 1:Nb
 
    for j = 1:N_W2
      fprintf(fid, 'W%d(%d,%d) = %1.14e \n',counter, i, j, W(i,j))
    end
  end

  fprintf(fid, '\n')
  fprintf(fid, '! ------- \n')
  fprintf(fid, '\n')

end

fprintf(fid, 'ATTENTION! MATRICES ARE WRITTEN IN SINGLE PRECISION! SEE THE "e" SYMBOL? CONVERT MANUALLY WITH SUBSTITUTE ALL!\n')
fprintf(fid, 'ATTENTION! MATRICES ARE WRITTEN IN SINGLE PRECISION! SEE THE "e" SYMBOL? CONVERT MANUALLY WITH SUBSTITUTE ALL!\n')
fprintf(fid, 'ATTENTION! MATRICES ARE WRITTEN IN SINGLE PRECISION! SEE THE "e" SYMBOL? CONVERT MANUALLY WITH SUBSTITUTE ALL!\n')

fclose(fid);
 
fprintf('Model exported into "exported_model.f90"\n\n')


