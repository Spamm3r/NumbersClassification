import numpy as np
import mnist


class NeuralNetwork:

    def __init__(self):
        self.x_train, self.t_train, self.x_test, self.t_test = mnist.load()
        self.x_train, self.t_train = self.x_train / 255.0, self.t_train / 255.0
        print(self.x_train[0])
        # y = np.empty((28, 28))
        # k = 0
        # for i in range(0, 28):
        #     for j in range(0, 28):
        #         y[i][j] = self.sigmoid(self.x_train[0][k])
        #         k += 1
        # for i in range(0, len(self.x_train[0])):
        #     print('{0:.16f}'.format(self.sigmoid(self.x_train[0][i])))


    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))


if __name__ == "__main__":
    mnist.init()
    nn = NeuralNetwork()
