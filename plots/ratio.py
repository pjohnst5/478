import numpy as np
import matplotlib.pyplot as plt

n_groups = 6

bpNoMomentum = (0.4121212121, 0.4477272727, 0.3977272727, 0.5257575758, 0.403030303, 0.5090909091)
bpMomentum30 =  (0.4325757576, 0.4833333333, 0.5204545455, 0.4924242424, 0.6356060606, 0.4734848485)
bpMomentum15 = (0.4143939394, 0.4886363636, 0.4909090909, 0.5257575758, 0.5257575758, 0.5265151515)

fig, ax = plt.subplots()
index = np.arange(n_groups)
bar_width = 0.15
opacity = 0.8
font_size_large = 16
font_size_small = 13

rects1 = plt.bar(index, bpNoMomentum, bar_width,
alpha=opacity,
color='indigo',
label='No Momentum, Interval of 30')

rects2 = plt.bar(index + bar_width, bpMomentum30, bar_width,
alpha=opacity,
color='crimson',
label='Momentum of 0.9, Interval of 30')

rects3 = plt.bar(index + bar_width*2, bpMomentum15, bar_width,
alpha=opacity,
color='mediumturquoise',
label='Momentum of 0.9, Interval of 15')

plt.ylim(0,1.0)
plt.xlabel('Number of Hidden Nodes', fontsize=font_size_large)
plt.ylabel('Classification Accuracy', fontsize=font_size_large)
plt.title('Ratios dataset using MLP with Backpropagation\n', fontweight='bold', fontsize=font_size_small)
plt.xticks(index + bar_width, ('2', '4', '8', '16', '32', '64'), fontsize=font_size_large)
plt.yticks(fontsize=font_size_small)
plt.legend(fontsize=font_size_small)


plt.tight_layout()
plt.show()
