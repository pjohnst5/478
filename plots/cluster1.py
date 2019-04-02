import matplotlib.pyplot as plt
import pandas as pd

k = [2,3,4,5,6,7]
sses = [12.144,
        6.998,
        5.533,
        5.86,
        3.946,
        4.209]

df=pd.DataFrame({'x': k, 'y1': sses })

plt.title("Iris K clusters vs SSE", fontweight='bold')
plt.xlabel("K values")
plt.ylabel("SSE")

plt.plot( 'x', 'y1', data=df, marker='', color='red', linewidth=2)
plt.show()
