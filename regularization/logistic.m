clear all; close all; clc;

x = load('ex5Logx.dat');
y = load('ex5Logy.dat');

figure

% Find the indices for the 2 classes.
pos = find(y); neg = find(y == 0);

plot(x(pos, 1), x(pos, 2), '3+');
hold on;
plot(x(neg, 1), x(neg, 2), '1o');
legend('y = 1', 'y = 0');

MAX_ITER = 20;

m = length(x);
n = 28;
lambda = 1;
lambda_diagonal = diag(ones(m, 1), n, n);
lambda_diagonal(1, 1) = 0;
x = map_feature(x(:, 1), x(:, 2));
theta = zeros(n, 1);
g = inline('1.0 ./ (1.0 + exp(-z))');
J = zeros(MAX_ITER, 1);

for i = 1:MAX_ITER
    h = g(x * theta);
    regularization = (lambda / (2 * m)) * sum(theta(2:n, 1) .^ 2);
    J(i) = (1 / m) * sum(-y .* log(h) - (1 - y) .* log(1 - h)) + regularization;

    % Add a zero to the intercept term of theta, so that we dont try to regularize it.
    % This will cause the (lambda / m) .* theta term below to be zero for the intercept
    % term.
    theta_zero_intercept = theta(:, :);
    theta_zero_intercept(1) = 0;
    grad = (1 / m) .* x' * (h - y) + (lambda / m) .* theta_zero_intercept;

    H = (1 / m) .* x' * diag(h) * diag(1 - h) * x + (lambda / m) * lambda_diagonal;
    theta = theta - H \ grad;
endfor

% Define the ranges of the grid
u = linspace(-1, 1.5, 200);
v = linspace(-1, 1.5, 200);

% Initialize space for the values to be plotted
z = zeros(length(u), length(v));

% Evaluate z = theta*x over the grid
for i = 1:length(u)
    for j = 1:length(v)
        % Notice the order of j, i here!
        z(j,i) = map_feature(u(i), v(j))*theta;
    end
end

% Because of the way that contour plotting works
% in Matlab, we need to transpose z, or
% else the axis orientation will be flipped!
z = z';
% Plot z = 0 by specifying the range [0, 0]
contour(u,v,z, [0, 0], 'LineWidth', 2);