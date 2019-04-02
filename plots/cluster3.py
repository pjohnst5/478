import matplotlib.pyplot as plt
import pandas as pd

test = [1,2,3,4,5]

sses = [6.856,
        6.857,
        6.622,
        6.301,
        6.62]

df=pd.DataFrame({'x': test, 'y1': sses })

plt.title("Iris k=4 (ouput label used as feature) test number vs SSE", fontweight='bold')
plt.xlabel("Test number")
plt.ylabel("SSE")

plt.plot( 'x', 'y1', data=df, marker='', color='blue', linewidth=2)
plt.show()
