clear all; close all; clc;

x = load('ex4x.dat');
y = load('ex4y.dat');

MAX_ITER = 5;

features = 2;
n = features + 1;
m = length(x);

x = [ones(m, 1), x];
pos = find(y == 1);
neg = find(y == 0);

figure;
plot(x(pos, 2), x(pos, 3), '+');
hold on;
plot(x(neg, 2), x(neg, 3), 'o');

theta = zeros(n, 1);
g = inline('1.0 ./ (1.0 + exp(-z))');

J = zeros(MAX_ITER, 1);

for i = 1:MAX_ITER
    h = g(x * theta);
    J(i) = (1 / m) * sum(-y .* log(h) - (1 - y) .* log(1 - h));

    grad = (1 / m) .* x' * (h - y);
    H = (1 / m) .* x' * diag(h) * diag(1 - h) * x;

    theta = theta - H \ grad;
endfor

% Plot J of theta.
% figure
% plot((1:MAX_ITER), J);

% Draw decision boundary.
plot_x = [min(x(:,2)) - 2,  max(x(:,2)) + 2];
plot_y = (-1 ./ theta(3)) .* (theta(2) .* plot_x + theta(1));
plot(plot_x, plot_y);

% Make prediction using values of theta found.
results = 1 - g(theta' * [1; 20; 80])
% 66% chance of not being admitted.
