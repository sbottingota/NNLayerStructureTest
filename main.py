from sklearn.neural_network import MLPClassifier
import pickle

def generatorToList(generator):
    output = []
    for item in generator:
        output.append(item)

    return output

layerSizes = []

for i in range(1, 5):
    layerSizes.append([1 for j in range(i)])
    layerSizes.append([512 for j in range(i)])
    layerSizes.append([1024 for j in range(i)])

    if i > 1:
        increasingLayer = [1024]
        for j in range(i - 1):
            increasingLayer.append(increasingLayer[j] / 2)

        decreasingLayer = [512]
        for j in range(i - 1):
            decreasingLayer.append(decreasingLayer[j - 1] * 2)

        layerSizes.append(generatorToList(increasingLayer))
        layerSizes.append(generatorToList(decreasingLayer))

layerSizes.append([4096])

layerSizes.append([1 for i in range(10)])
layerSizes.append([128 for i in range(10)])

print(layerSizes)
