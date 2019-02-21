import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import numpy as np
import pandas as pd


numHiddenNodes = [1, 2, 4, 8, 16, 32, 64, 128]
MSETrains = [0.9091490629560638, 0.8282069026109673, 0.6868628059513203, 0.6638906424360174, 0.5391060581068156, 0.5302168942704489, 0.5298351043722962, 0.5228513015914886]
MSEValids = [0.9101645900693488, 0.8424954088371828, 0.6914963405988737, 0.7001922714697806, 0.5477065843849064, 0.596744359849076, 0.5642501649319848, 0.5843557904396536]
MSETests = [0.9111137799733948, 0.8384954564406845, 0.7158285879547204, 0.7268938796458813, 0.5606684613901225, 0.6041098426336292, 0.6084156131261841, 0.5907740081698295]
testingSetAccuracies = [0.06451612903225806, 0.18548387096774194, 0.4475806451612903, 0.4112903225806452, 0.6693548387096774, 0.5685483870967742, 0.5645161290322581, 0.5846774193548387]

df=pd.DataFrame({'x': numHiddenNodes, 'y1': MSETrains, 'y2': MSEValids, 'y3': MSETests })

plt.title("Vowel MSE vs. # Hidden Nodes", fontweight='bold')
plt.xlabel("# Hidden Nodes")
plt.ylabel("MSE")

plt.plot( 'x', 'y1', data=df, marker='', color='skyblue', linewidth=2, label="MSE of TRS")
plt.plot( 'x', 'y2', data=df, marker='', color='black', linewidth=2, label="MSE of VS")
plt.plot( 'x', 'y3', data=df, marker='', color='red', linewidth=2, label="MSE of TS")
plt.legend()


plt.show()
