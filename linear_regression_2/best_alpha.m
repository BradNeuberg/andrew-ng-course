clear all; close all; clc

% The number of features our input data has.
n = 2;

x = load('ex3x.dat');
y = load('ex3y.dat');

m = length(x);

x = [ones(m, 1), x];

% Scale the input values to be closer in value to make gradient descent quicker.
sigma = std(x);
avg = mean(x)
original_x = x;
x(:, 2) = (x(:, 2) - avg(2)) ./ sigma(2);
x(:, 3) = (x(:, 3) - avg(3)) ./ sigma(3);

% The best alpha that was found.
alpha = 1.0;

theta = zeros(n + 1, 1);
for i = 1:100
    gradient = (1 / m) * x' * (x * theta - y);
    theta = theta - alpha * gradient;
endfor

input_size = (1650 - avg(2)) / sigma(2)
input_bedrooms = (3 - avg(3)) / sigma(3)
predicted_price = [1, input_size, input_bedrooms] * theta