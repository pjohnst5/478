import matplotlib.pyplot as plt
import pandas as pd

#no distance weighting, normalization
k = [1,3,5,7,9,11,13,15]
accuracies = [0.8159420289855074,
              0.8347826086956521,
              0.8405797101449275,
              0.8536231884057971,
              0.8579710144927537,
              0.8521739130434783,
              0.8565217391304347,
              0.8521739130434783]

df=pd.DataFrame({'x': k, 'y1': accuracies })

plt.title("Credit - Weighted\nClassification accuracy (cross fold mean) vs. K values", fontweight='bold')
plt.xlabel("K values")
plt.ylabel("Accuracy")

plt.plot( 'x', 'y1', data=df, marker='', color='blue', linewidth=2)
plt.show()
