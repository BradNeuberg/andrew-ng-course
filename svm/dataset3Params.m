function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

C_try = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];
sigma_try = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];

% Store the best C and sigma combination so far given a prediction error rate on our
% cross validation data set.
prediction_error = -1;
C = -1;
sigma = -1;

% Compute model using each combination of hyperparameters.
for i = 1:size(C_try, 1)
  for j = 1:size(sigma_try, 1)
    this_C = C_try(i);
    this_sigma = sigma_try(j);
    model = svmTrain(X, y, this_C, @(x1, x2) gaussianKernel(x1, x2, this_sigma));

    % For each model that is fit, compute predictions based on the cross validation set.
    predictions = svmPredict(model, Xval);

    % Compute the prediction error and store it if it is better than what we've encountered so far.
    this_prediction_error = mean(double(predictions ~= yval));

    fprintf('Using C = %f and sigma = %f, prediction error on validation is = %f\n',
          this_C, this_sigma, this_prediction_error);

    if prediction_error == -1 || this_prediction_error < prediction_error
      prediction_error = this_prediction_error;
      C = this_C;
      sigma = this_sigma;
    endif
  end
end

fprintf('Best C value found: %f\n', C);
fprintf('Best sigma value found: %f\n', sigma);

% =========================================================================

end
