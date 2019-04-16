import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


epochs = []
trainingLoss = []

df=pd.DataFrame({'x': epochs, 'y1': trainingLoss})

plt.title("Training Loss vs. Epochs", fontweight='bold')
plt.xlabel("Epochs")
plt.ylabel("Training Loss")

plt.plot( 'x', 'y1', data=df, marker='', color='blue', linewidth=2)
plt.legend()
plt.show()
