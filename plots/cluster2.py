import matplotlib.pyplot as plt
import pandas as pd

k = [2,3,4,5,6,7]

sses = [62.144,
         7.817,
         6.614,
         5.652,
         4.766,
         6.397]

df=pd.DataFrame({'x': k, 'y1': sses })

plt.title("Iris (ouput label used as feature) K clusters vs SSE", fontweight='bold')
plt.xlabel("K values")
plt.ylabel("SSE")

plt.plot( 'x', 'y1', data=df, marker='', color='green', linewidth=2)
plt.show()
