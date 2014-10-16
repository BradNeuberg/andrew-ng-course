clear all; close all; clc

x = load('ex2x.dat')
y = load('ex2y.dat')

figure;
plot(x, y, 'o');
ylabel('Height in meters');
xlabel('Age in years');

m = length(x);
x = [ones(m, 1), x];
n = 1;
alpha = 0.07;
theta = zeros(n + 1, 1);

for i = 1:1500
    gradient = (1 / m) * x' * (x * theta - y);
    theta = theta - alpha * gradient;
endfor

hold on;
plot(x(:, 2), x * theta, '-');
legend('Training data', 'Linear regression');