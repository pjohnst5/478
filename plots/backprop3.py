import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import numpy as np
import pandas as pd


learningRates = [0.005, 0.025, 0.05, 0.075, 0.085, 0.1, 0.125, 0.15, 0.175, 0.2]
totalEpochs = [572, 145 ,63, 86, 68 ,85, 39, 43, 35, 31]
testSetAccuracy = [0.5080645161290323, 0.5403225806451613 ,0.5766129032258065, 0.625, 0.6491935483870968 ,0.6451612903225806, 0.532258064516129, 0.6290322580645161, 0.5483870967741935, 0.5403225806451613]

df=pd.DataFrame({'x': learningRates, 'y1': totalEpochs})

plt.title("Epochs for best VS solution vs. Learning Rates", fontweight='bold')
plt.xlabel("Learning Rate")
plt.ylabel("Epochs")

plt.plot( 'x', 'y1', data=df, marker='', color='blue', linewidth=2)
plt.legend()


plt.show()
