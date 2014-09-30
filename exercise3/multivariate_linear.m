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
x(:, 2) = (x(:, 2) - avg(2)) ./ sigma(2);
x(:, 3) = (x(:, 3) - avg(3)) ./ sigma(3);

% Potential alpha learning rate values to test, in rough increments of *0.3.
potential_alphas = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 1.3, 2.0, 2.3, 3.0, 3.3, 4.0, 4.3, 5.0, 5.3, 6.0, 6.3, 7.0, 7.3, 8.0, 8.3, 9.0, 9.3, 10.0];
%potential_alphas = [0.01, 0.03, 0.1, 0.3, 1, 1.3];
J = zeros(length(potential_alphas), 50);

for alpha_idx = 1:length(potential_alphas)
    alpha = potential_alphas(alpha_idx)
    theta = zeros(n + 1, 1);
    for i = 1:50
        J(alpha_idx, i) = (1 / (2 * m)) * (x * theta - y)' * (x * theta - y)
        gradient = (1 / m) * x' * (x * theta - y);
        theta = theta - alpha * gradient;
    endfor
endfor

figure;
xlabel('Iterations');
ylabel('Cost J');

hold on;
plot(0:49, J(1, 1:50), '-');
plot(0:49, J(2, 1:50), 'b-');
plot(0:49, J(3, 1:50), 'r-');
plot(0:49, J(4, 1:50), 'k-');
plot(0:49, J(5, 1:50), '--');
plot(0:49, J(6, 1:50), 'b--');
plot(0:49, J(7, 1:50), 'r--');
plot(0:49, J(8, 1:50), 'k--');
plot(0:49, J(9, 1:50), 'b-');
plot(0:49, J(10, 1:50), 'r-');
plot(0:49, J(11, 1:50), 'k-');
plot(0:49, J(12, 1:50), 'bd');
plot(0:49, J(13, 1:50), 'rd');
plot(0:49, J(14, 1:50), 'kd');
plot(0:49, J(15, 1:50), 'b.');
plot(0:49, J(16, 1:50), 'r.');
plot(0:49, J(17, 1:50), 'k.');
plot(0:49, J(18, 1:50), 'b--');
plot(0:49, J(19, 1:50), 'r--');
plot(0:49, J(20, 1:50), 'k--');
plot(0:49, J(21, 1:50), 'b+');
plot(0:49, J(22, 1:50), 'r+');
plot(0:49, J(23, 1:50), 'k+');
plot(0:49, J(24, 1:50), 'b:');
plot(0:49, J(25, 1:50), 'r:');
