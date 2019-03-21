import matplotlib.pyplot as plt
import pandas as pd

#no distance weighting, normalization
k = [1,3,5,7,9,11,13,15]
mses = [23.580980392156874,
        14.215446623093682,
        16.31825882352942,
        20.028679471788717,
        20.14123456790124,
        21.274496840058337,
        22.408194686158495,
        22.58496470588235]

df=pd.DataFrame({'x': k, 'y1': mses })

plt.title("Housing - Unweighted\nMSE vs. K values", fontweight='bold')
plt.xlabel("K values")
plt.ylabel("MSE")

plt.plot( 'x', 'y1', data=df, marker='', color='red', linewidth=2)
plt.show()
