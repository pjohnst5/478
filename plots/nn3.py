import matplotlib.pyplot as plt
import pandas as pd

#no distance weighting, normalization
k = [1,3,5,7,9,11,13,15]
accuracies = [0.7874287428742874,
              0.8156315631563157,
              0.8267326732673267,
              0.8297329732973298,
              0.8291329132913291,
              0.830933093309331,
              0.8324332433243324,
              0.8315331533153315]

df=pd.DataFrame({'x': k, 'y1': accuracies })

plt.title("Magic telescope - Weighted\nClassification accuracy vs. K values", fontweight='bold')
plt.xlabel("K values")
plt.ylabel("Accuracy")

plt.plot( 'x', 'y1', data=df, marker='', color='green', linewidth=2)
plt.show()
