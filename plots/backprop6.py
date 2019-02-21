import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import numpy as np
import pandas as pd


numHiddenLayers = [1, 2, 3, 4]
MSETrains = [0.8375367688398625, 0.7351189602223023, 0.9095731198971241, 0.9096725836537702]
MSEValids = [0.8395150113083117, 0.7501013827899274, 0.9098114843096038, 0.910878446260575]
MSETests = [0.8382917655743833, 0.7658645709805685, 0.9114722818019874, 0.9121184591510654]
testingSetAccuracies = [0.20967741935483872, 0.33064516129032256, 0.056451612903225805, 0.06451612903225806]
epochs = [45, 107, 17, 7]

df=pd.DataFrame({'x': numHiddenLayers, 'y1': MSETrains, 'y2': MSEValids, 'y3': MSETests, 'y4': testingSetAccuracies })

plt.title("Vowel MSE and Accuracy vs. # Hidden Layers", fontweight='bold')
plt.xlabel("# Hidden Layers (of size 4)")
plt.ylabel("MSE / Accuracy")

plt.plot( 'x', 'y1', data=df, marker='', color='skyblue', linewidth=2, label="MSE of TRS")
plt.plot( 'x', 'y2', data=df, marker='', color='black', linewidth=2, label="MSE of VS")
plt.plot( 'x', 'y3', data=df, marker='', color='orange', linewidth=2, label="MSE of TS")
plt.plot(  'x', 'y4', data=df, marker='', color='red', linewidth=2, label="Testing Set Accuracy")
plt.legend()


plt.show()
