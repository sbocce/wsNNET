function [output1, output2] = apply_saved_net(model, activations, in_vect)

% Init output
output = in_vect;

% Loop through the layers of the neural net
counter = 0;
for [val, key] = model.model_weights

  % +++++++++++ Layer number +++++++++++++

  counter = counter + 1;

  % ++++++++++ Extract weights matrix and biases ++++++++++++++

  s = getfield(val, key); % extract the two fields at this layer
  
  W = s.kernel_0;
  b = s.bias_0';

  % +++++++++++ Apply the layer ++++++++++
  if strcmp(activations{counter}, 'tanh')
    output = tanh(W*output + b);
  elseif strcmp(activations{counter}, 'linear')
    output = W*output + b;
  else
    error('ATTENTION! Layers exceed elements in array of activation functions!')
  end

  output1 = output(1);
  output2 = output(2);

end


