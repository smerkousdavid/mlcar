from mlcar.neuralnet.network import NeuralNetwork
from mlcar.neuralnet.genetic import *

n = NeuralNetwork(2, 5, 1)
n.generate()

b = [
    TwinBreed(0),
    TwinBreed(1),
    ChampionBreed(0, 1),
    ChampionBreed(1, 2)
]

e = Evolution(b, n)
e.create()

e.load("test_evolution.weights")
# n.generate()

print("Prediction")
print(e.run(1, 2))

# e.save("test_evolution.weights")
