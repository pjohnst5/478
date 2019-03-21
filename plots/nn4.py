import matplotlib.pyplot as plt
import pandas as pd

#no distance weighting, normalization
k = [1,3,5,7,9,11,13,15]
mses = [23.58098039215688,
        12.310558867751297,
        11.179088494440387,
        11.073892657897746,
        11.736722469473253,
        12.169079321844473,
        12.598757816257997,
        12.460864170122159]

df=pd.DataFrame({'x': k, 'y1': mses })

plt.title("Housing - Weighted\nMSE vs. K values", fontweight='bold')
plt.xlabel("K values")
plt.ylabel("MSE")

plt.plot( 'x', 'y1', data=df, marker='', color='red', linewidth=2)
plt.show()
