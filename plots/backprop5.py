import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import numpy as np
import pandas as pd

momentumTerms = [0.0125, 0.025, 0.0375, 0.05, 0.075, 0.1, 0.125]
epochs = [46, 58, 84, 72, 54, 50, 42]
testingSetAccuracies = [0.5161290322580645, 0.5564516129032258, 0.625, 0.532258064516129, 0.5362903225806451, 0.5887096774193549, 0.45161290322580644]

df=pd.DataFrame({'x': momentumTerms, 'y1': epochs })

plt.title("Vowel Epochs vs. Momentum", fontweight='bold')
plt.xlabel("Momentum Term")
plt.ylabel("Epochs")

plt.plot( 'x', 'y1', data=df, marker='', color='green', linewidth=2)
plt.legend()


plt.show()
