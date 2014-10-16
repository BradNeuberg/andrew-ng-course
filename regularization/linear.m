

x = load('ex5Linx.dat');
y = load('ex5Liny.dat');

n = 5;
m = length(x);
lambda = 10;
lambda_diagonal = diag(ones(m, 1), n + 1, n + 1);
lambda_diagonal(1, 1) = 0;

x = [ones(m, 1), x, x.^2, x.^3, x.^4, x.^5];

theta = inv(x' * x + lambda * lambda_diagonal) * x' * y;

figure;
hold on;
plot(x(:, 2), y, '1*');
plot(x(:, 2), x * theta, 's');
legend(strcat('Lambda: ', num2str(lambda)));