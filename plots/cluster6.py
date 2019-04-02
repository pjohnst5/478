import matplotlib.pyplot as plt
import pandas as pd

ks = [2,3,4,5,6,7]

sses = [2650.346,
        1493.506,
        1077.612,
        901.539,
        700.109,
        617.575]

df=pd.DataFrame({'x': ks, 'y1': sses })

plt.title("Abalone no normalization (ouput label used as feature) K vs SSE", fontweight='bold')
plt.xlabel("K")
plt.ylabel("SSE")

plt.plot( 'x', 'y1', data=df, marker='', color='orange', linewidth=2)
plt.show()
