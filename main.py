from sklearn.neural_network import MLPClassifier
from keras.datasets import mnist
import numpy as np

import pickle
import itertools

layerSizes = []

for i in range(1, 5):
    layerSizes.append([1 for j in range(i)])
    layerSizes.append([512 for j in range(i)])
    layerSizes.append([1024 for j in range(i)])

    if i > 1:
        increasingLayer = [512]
        for j in range(i - 1):
            increasingLayer.append(increasingLayer[j] * 2)

        decreasingLayer = [1024]
        for j in range(i - 1):
            decreasingLayer.append(decreasingLayer[j] // 2)

        layerSizes.append(increasingLayer)
        layerSizes.append(decreasingLayer)

layerSizes.append([4096])

layerSizes.append([1 for i in range(10)])
layerSizes.append([128 for i in range(10)])

(xTrain, yTrain), (xTest, yTest) = mnist.load_data()

xTrain = np.reshape(xTrain, (len(xTrain), 28 ** 2))
xTest = np.reshape(xTest, (len(xTest), 28 ** 2))

for layerSize in layerSizes:
    clf = MLPClassifier(hidden_layer_sizes=layerSize, learning_rate="invscaling", verbose=False, max_iter=200)
    clf.fit(xTrain, yTrain)

    predictions = clf.predict(xTest)

    nRight = nWrong = 0

    for i in range(len(predictions)):
        if predictions[i] == yTest[i]:
            nRight += 1

        else:
            nWrong += 1

    print("Architecture:", layerSize)
    print("Accuracy:", nRight / (nRight + nWrong))

    nWeights = len(list(itertools.chain(*list(itertools.chain(*clf.coefs_)))))
    nBiases = len(list(itertools.chain(*clf.coefs_)))
    params = nWeights + nBiases

    print("Parameters:", params)

    pickle.dump(clf, open("networks/" + str(layerSize) + ".pkl", "wb"))
