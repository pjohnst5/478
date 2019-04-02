import matplotlib.pyplot as plt
import pandas as pd

ks = [2,3,4,5,6,7]

silhouette = [0.341,
              0.552,
              0.525,
              0.491,
              0.459,
              0.434]

df=pd.DataFrame({'x': ks, 'y1': silhouette })

plt.title("Abalone normalization (ouput label used as feature) K vs Avg. Silhouette", fontweight='bold')
plt.xlabel("K")
plt.ylabel("Avg. Silhouette")

plt.plot( 'x', 'y1', data=df, marker='', color='cyan', linewidth=2)
plt.show()
