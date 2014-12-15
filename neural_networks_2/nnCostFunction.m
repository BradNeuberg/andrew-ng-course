function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%


% Feed the values forward.
A_3 = zeros(m, num_labels);
for i = 1:m
  z_2 = Theta1 * [1; X(i, :)'];
  a_2 = sigmoid(z_2);

  z_3 = Theta2 * [1; a_2];
  A_3(i, :) = sigmoid(z_3)';
endfor

% Turn the correct y values into 0 or 1 based on them being present.
Y = zeros(m, num_labels);
for i = 1:m
  Y(i, y(i)) = 1;
endfor

% Now compute the cost given these theta and activation values.
for i = 1:m
  for k = 1:num_labels
    expected = Y(i, k);
    actual = A_3(i, k);
    J += ((-expected * log(actual)) - (1 - expected) * log(1 - actual));
  endfor
endfor
J = (1 / m) * J;

% Regularize the cost function.
regularize = 0;
Theta1_no_bias = Theta1(:, 2:end);
Theta2_no_bias = Theta2(:, 2:end);
for j = 1:hidden_layer_size
  for k = 1:input_layer_size
    regularize += Theta1_no_bias(j, k) ^ 2;
  endfor
endfor
for j = 1:num_labels
  for k = 1:hidden_layer_size
    regularize += Theta2_no_bias(j, k) ^ 2;
  endfor
endfor
regularize = (lambda / (2 * m)) * regularize;
J += regularize;

% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%

delta_1 = zeros(hidden_layer_size, input_layer_size + 1);
delta_2 = zeros(num_labels, hidden_layer_size + 1);
for t = 1:m
  a_1 = [1; X(t, :)'];

  z_2 = Theta1 * a_1;
  a_2 = [1; sigmoid(z_2)];

  z_3 = Theta2 * a_2;
  a_3 = sigmoid(z_3);

  error_3 = a_3 - Y(t, :)';
  error_2 = Theta2' * error_3 .* [1; sigmoidGradient(z_2)];

  delta_1 = delta_1 + error_2(2:end) * a_1';
  delta_2 = delta_2 + error_3 * a_2';
endfor

Theta1_grad = (1 / m) * delta_1;
Theta2_grad = (1 / m) * delta_2;

% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

Theta1_grad = Theta1_grad + [zeros(hidden_layer_size, 1), (lambda / m) * Theta1(:, 2:end)];
Theta2_grad = Theta2_grad + [zeros(num_labels, 1), (lambda / m) * [zeros(0), Theta2(:, 2:end)]];




















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
