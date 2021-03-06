from fc_layer import FCLayer
from activation_layer import ActivationLayer
from activation_functions import ActivationFunctions
from loss_functions import LossFunctions
from network import Network
import numpy as np




if __name__ == '__main__':
    xTrain = np.array([[[0, 0]], [[0, 1]], [[1, 0]], [[1, 1]]])
    yTrain = np.array([[[0]], [[1]], [[1]], [[0]]])

    network = Network()
    network.add(FCLayer(2, 3))
    network.add(ActivationLayer(ActivationFunctions.tanh, ActivationFunctions.tanhPrime))
    network.add(FCLayer(3, 1))
    network.add(ActivationLayer(ActivationFunctions.tanh, ActivationFunctions.tanhPrime))

    network.setLoss(LossFunctions.mse, LossFunctions.msePrime)
    network.fit(xTrain, yTrain, 1000, 0.3, showLogs = False)
        
    print(list(map(lambda i : round(i[0][0]), network.predict(xTrain))))
