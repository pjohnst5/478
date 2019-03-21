import matplotlib.pyplot as plt
import pandas as pd

#no distance weighting, normalization
k = [1,3,5,7,9,11,13,15]
accuracies = [0.7874287428742874,
              0.8157815781578158,
              0.8271827182718272,
              0.8294329432943295,
              0.8282328232823283,
              0.8294329432943295,
              0.8297329732973298,
              0.8304830483048304 ]

df=pd.DataFrame({'x': k, 'y1': accuracies })

plt.title("Magic telescope - Unweighted\nClassification accuracy vs. K values", fontweight='bold')
plt.xlabel("K values")
plt.ylabel("Accuracy")

plt.plot( 'x', 'y1', data=df, marker='', color='green', linewidth=2)
plt.show()
