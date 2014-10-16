clear all; close all; clc

% The number of features our input data has.
n = 2;

x = load('ex3x.dat');
y = load('ex3y.dat');

m = length(x);

x = [ones(m, 1), x];

theta = inv(x' * x) * x' * y

predicted_price = [1, 1650, 3] * theta;
