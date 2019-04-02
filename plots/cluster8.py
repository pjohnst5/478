import matplotlib.pyplot as plt
import pandas as pd

minDistances = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1]

silhouette = [0.486,
              0.485,
              0.511,
              0.486,
              0.525,
              0.497,
              0.511,
              0.486,
              0.497,
              0.511,
              0.497,
              0.511,
              0.486,
              0.486,
              0.511,
              0.522,
              0.486,
              0.511,
              0.525,
              0.486]

df=pd.DataFrame({'x': minDistances, 'y1': silhouette })

plt.title("Abalone normalization (ouput label used as feature) Min. Initial Centroid Distance vs Avg. Silhouette", fontweight='bold')
plt.xlabel("Minimum Distance")
plt.ylabel("Avg. Silhouette")

plt.plot( 'x', 'y1', data=df, marker='', color='cobalt', linewidth=2)
plt.show()
