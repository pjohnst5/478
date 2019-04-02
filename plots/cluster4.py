import matplotlib.pyplot as plt
import pandas as pd

ks = [2,3,4,5,6,7]

sses = [281.301,
        95.211,
        81.602,
        78.522,
        52.722,
        73.293]

df=pd.DataFrame({'x': ks, 'y1': sses })

plt.title("Abalone normalization (ouput label used as feature) K vs SSE", fontweight='bold')
plt.xlabel("K")
plt.ylabel("SSE")

plt.plot( 'x', 'y1', data=df, marker='', color='purple', linewidth=2)
plt.show()
