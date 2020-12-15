import numpy as np
import sys


def sigmoid(x): return 1 / (1 + np.exp(-x))


def softmax(x): return np.exp(x) / np.sum(np.exp(x))


def get_features_minmax_values(x):
    min_list = x.min(axis=0)
    max_list = x.max(axis=0)
    return min_list, max_list


def minmax_norm(x, min_x, max_x):
    # Normalizing each example in training set.
    for i in range(len(x[0])):
        if min_x[i] == max_x[i]:
            x[:, i] = 0
        else:
            x[:, i] = (np.subtract(x[:, i], min_x[i])/(max_x[i] - min_x[i]))


def minmax_norm_vector(x, min_x, max_x):
    for i in range(len(x)):
        if min_x[i] == max_x[i]:
            x[i] = 0
        else:
            x[i] = (x[i] - min_x[i])/(max_x[i] - min_x[i])


def instantiate_variables():
    x, y = np.loadtxt(sys.argv[1], max_rows=1000), np.loadtxt(sys.argv[2], max_rows=1000)
    min_x, max_x = get_features_minmax_values(x)
    minmax_norm(x, min_x, max_x)
    w1 = np.random.rand(300, len(x[0]))
    b1 = np.zeros(300)
    w2 = np.random.rand(10, 300)
    b2 = np.zeros(10)
    return {'x': x, 'y': y, 'w1': w1, 'b1': b1, 'w2': w2, 'b2': b2, 'min_x': min_x, 'max_x': max_x}


def front_prop(x, y, params):
    w1, w2, b1, b2 = [params[key] for key in ('w1', 'w2', 'b1', 'b2')]
    z1 = np.dot(w1, x) + b1
    minmax_norm_vector(z1, params['min_x'], params['max_x'])  # TODO check if this is responsible for bad success rate.
    h1 = sigmoid(z1)
    z2 = np.dot(w2, h1) + b2
    h2 = softmax(z2)
    loss = -(np.log(h2[y]))
    ret = {'x': x, 'y': y, 'z1': z1, 'z2': z2, 'h1': h1, 'h2': h2, 'loss': loss}
    for key in params:
        ret[key] = params[key]
    return ret


def back_prop(fprop_cache):
    x, y, z1, h1, z2, h2, loss = [fprop_cache[key] for key in ('x', 'y', 'z1', 'h1', 'z2', 'h2', 'loss')]


def training_process():
    # TODO - increase number of rows.
    epochs, eta = 10, 0.01
    params = instantiate_variables()
    tr_set = list(zip(params['x'], params['y'].astype(int)))
    for i in range(epochs):
        np.random.shuffle(tr_set)
        for x, y in tr_set:
            # forward through neural network and calculate the loss.
            front_prop(x, y, params)  # TODO
            # compute gradients.
            back_prop()  # TODO


if __name__ == '__main__':
    training_process()
    # output_predictions()
