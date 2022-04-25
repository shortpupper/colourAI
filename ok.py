import random
import json
from random import randrange as rng

def load_weights():
 with open(r"E:\Python\prog\how\ai-stouf\waeight-new.json") as f: return json.loads(f.read())


colour_rgb = (255, 43, 0)

learning_rate = 0.0001 #0.000001
#bias = 1
roundToInt = 7
threshold = 1
whenModNodeHowManyTimes = 1


weightsDict = load_weights()
bias = weightsDict["bias"]

weightsDict.pop("bias")

weights = [list(weightsDict.items())[i][1:][0] for i in range(len(weightsDict.keys()))]

# constants
CARV = "+-"

# function to save the weights to a json file
def save_weights(bias, weights):
 data = {"bias": bias}
 for i in range(len(weights)): data[f"hidden{i}"] = weights[i]
 with open(r"E:\Python\prog\how\ai-stouf\waeight-new.json", "w") as f: f.write(str(json.dumps(data)))

# function to get change
def getChange(current, previous):
 if current == previous: return 0
 try: return (abs(current - previous) / previous) * 100.0 if current > previous else float('-'+str((abs(current - previous) / previous) * 100.0))
 except ZeroDivisionError: raise ValueError("yo you failed lol")


# function to make a perceptron 
def perceptron(bias: int, inputs: list[float], weights: list[float]) -> float: return round(sum([x*y+bias*y for (x, y) in zip(inputs, weights)]), roundToInt)


#print(perceptron(bias, [True, False, True,], [0.5, 0.2, 0.7,]))


# function to make a list if one of the rgb values is 0 or greater
def Inputs(rgb: tuple[int, int, int]) -> list[float]:
 return [round(x/255, 3) for x in rgb]

#print(inputs(colour_rgb))



# function to make a hidden layer of perceptron
def hiddenlayer(bias:int, inputs:list[float], weights:list[list[float]]) -> list[float]:
 return [perceptron(bias, inputs, weights[i]) for i in range(len(weights))]

# function to make a list of random numbers with args as count and round n
def nodeMaker(count: int, roundTo: int) -> list[int]: return [round(random.random(), roundTo) for i in range(count)]

# function to make blank nodes
def blankNodeMaker(count: int) -> list[int]: return [0.0 for i in range(count)]

#print(nodeMaker(14, 7))
# function to make hiddenlayers from nodeMaker
def makeHiddenLayer(countHiddenLayers: int, nodeCount: int, roundTo: int):
 return [blankNodeMaker(nodeCount) for i in range(countHiddenLayers)]


"""hiddenlayerA = hiddenlayer(bias, Inputs(colour_rgb), weights[0])

hiddenlayerB = hiddenlayer(bias, hiddenlayerA, weights[1])

output = sum(hiddenlayerB)

print(output)

if output >= threshold: 
 print("I think you like this colour")
 print("am I wrong?")
else:
 print("I think you don't like this colour")
 print("am I wrong?")"""


# function to run a network
def network(bias: int, weights: list, inputs: tuple[int]) -> int:
 previousHiddenLayer = Inputs(inputs)
 for i in range(len(weights)): previousHiddenLayer = hiddenlayer(bias, previousHiddenLayer, weights[i])
 return round(sum(previousHiddenLayer), roundToInt)
  


# function to modify a random nodes weight
def modifyNode(weights: list[list[list[float]]], times: int) -> list[list[list[float]]]:
 for i in range(times):
  r1 = rng(0, len(weights))
  r2 = rng(0, len(weights[r1]))
  r3 = rng(0, len(weights[r1][r2]))
  if rng(0, 2): weights[r1][r2][r3] -= learning_rate
  else: weights[r1][r2][r3] += learning_rate
 return weights



# function to test the network for errors
# then it returns a float indicating the average of the errors
def errorCatcher(bias: int, weights: list, real: list[tuple[int]], fake: list[tuple[int]]) -> float: return sum([getChange(x, threshold) for x in [network(bias, weights, real[i]) for i in range(len(real))]]+[-getChange(x, threshold) for x in [network(bias, weights, fake[i]) for i in range(len(fake))]])/(len(real)+len(fake))


# function to for modifly the network weights and or bias
def networkMod(bias: float, weights: list) -> tuple[float, list]:
 what = rng(0, 3)
 # if what is 0
 if not what: weights = modifyNode(weights, whenModNodeHowManyTimes)
 # if what is 1
 elif what-1: bias = round(bias-float(CARV[rng(0,2)]+str(learning_rate)), roundToInt)
 # if what is 2
 else:
  weights = modifyNode(weights, whenModNodeHowManyTimes)
  bias = round(bias-float(CARV[rng(0,2)]+str(learning_rate)), roundToInt)
 return (bias, weights)
 


GoodColoursRgbList = [(32, 179, 152), (41, 189, 8), (255, 208, 0)]
BadColoursRgbList =  [(214, 17, 211), (0, 4, 240), (255, 0, 140)]

# function to train the network
def train(count: int):
 printEvery = 1_000
 bestWeight, bestBias = weights, bias
 for i in range(count):
  print(f"round {i}") if i % printEvery == 0 else None  
  net1Mod = networkMod(bestBias, bestWeight)
  net1 = errorCatcher(net1Mod[0], net1Mod[1], GoodColoursRgbList, BadColoursRgbList)
  net2Mod = networkMod(bestBias, bestWeight)
  net2 = errorCatcher(net2Mod[0], net2Mod[1], GoodColoursRgbList, BadColoursRgbList)
  net3Mod = networkMod(bestBias, bestWeight)
  net3 = errorCatcher(net3Mod[0], net3Mod[1], GoodColoursRgbList, BadColoursRgbList)
  net4Mod = networkMod(bestBias, bestWeight)
  net4 = errorCatcher(net4Mod[0], net4Mod[1], GoodColoursRgbList, BadColoursRgbList)
  net5Mod = networkMod(bestBias, bestWeight)
  net5 = errorCatcher(net5Mod[0], net5Mod[1], GoodColoursRgbList, BadColoursRgbList)
  nets = [net1, net2, net3, net4, net5]
  j = nets.index(max(nets))
  bestBias, bestWeight = [net1Mod, net2Mod, net3Mod, net4Mod, net5Mod][j]
  print(nets) if i % printEvery == 0 else None
  print(f"best network is {j}") if i % printEvery == 0 else None
 save_weights(bestBias, bestWeight)
 print("Done and saved.")



#train(2_000)

# function to tst a
def test():
 GoodColoursRgbList = [(32, 179, 152), (41, 189, 8), (255, 208, 0)]
 BadColoursRgbList =  [(214, 17, 211), (0, 4, 240), (255, 0, 140)]
 bestWeight, bestBias = weights, bias
 net1Mod = networkMod(bestBias, bestWeight)
 net1 = errorCatcher(net1Mod[0], net1Mod[1], GoodColoursRgbList, BadColoursRgbList)
 print(net1Mod)
 print(net1)

def testNet():
 GoodColoursRgbList = [(32, 179, 152), (41, 189, 8), (255, 208, 0)]
 BadColoursRgbList =  [(214, 17, 211), (0, 4, 240), (255, 0, 140)]
 for x in GoodColoursRgbList:
  c = network(bias, weights, x)
  print(f"good colour think:{c}")
  print("I think you like this colour") if c >= threshold  else print("I think you don't like this colour")
 for x in BadColoursRgbList:
  c = network(bias, weights, x)
  print(f"Bad colour think:{c}")
  print("I think you like this colour") if c >= threshold  else print("I think you don't like this colour")
 

#testNet()
"""
#what it thinks
ansewr = network(bias, weights, (255, 0, 140))

print(f"value it thinks its right: {ansewr}")

if ansewr >= threshold: print("I think you like this colour")
else: print("I think you don't like this colour")


"""



# ideas

# maby when the input is RGB in stead mkae it so it a per sent like 0.4 of the normal colour value in stead of 0 or 1
