import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

NUM_FEATURES = 1
NUM_ITERATIONS = 1500

def prepare_data():
    x = np.loadtxt('ex2x.dat')
    x = x.reshape(len(x), 1)
    x = np.insert(x, 0, values=1.0, axis=1)

    y = np.loadtxt('ex2y.dat')
    y = y.reshape(len(y), 1)

    return (x, y)

def batch_gradient_descent():
    x, y = prepare_data()
    draw_start_graph(x, y)

    m = len(x)
    alpha = 0.07
    theta = np.zeros((NUM_FEATURES + 1, 1))

    for i in range(NUM_ITERATIONS):
        gradient = (1.0 / m) * x.transpose().dot((x.dot(theta) - y))
        theta = theta - alpha * gradient

    print "Final results: " + np.array_str(theta)

    draw_end_graph(x, theta)

def draw_start_graph(x, y):
    plt.plot(x[:, 1], y, 'o')
    plt.xlabel('Age in years')
    plt.ylabel('Height in meters')
    plt.axis([0, np.amax(x[:, 1]), 0, np.amax(y)])

def draw_end_graph(x, theta):
    plt.plot(x[:, 1], x.dot(theta), '-')
    plt.show()

batch_gradient_descent()