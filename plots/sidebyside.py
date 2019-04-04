import numpy as np
import matplotlib.pyplot as plt

n_groups = 6

bpNoMomentum = (0.7234848485, 0.8310606061, 0.7477272727, 0.8424242424, 0.7704545455, 0.7886363636)
bpMomentum30 =  (0.8151515152, 0.8045454545, 0.8507575758, 0.7704545455, 0.8583333333, 0.8416666667)
bpMomentum15 = (0.8234848485, 0.8583333333, 0.8340909091, 0.7878787879, 0.7954545455, 0.8242424242)

fig, ax = plt.subplots()
index = np.arange(n_groups)
bar_width = 0.15
opacity = 0.8
font_size_large = 16
font_size_small = 13

rects1 = plt.bar(index, bpNoMomentum, bar_width,
alpha=opacity,
color='steelblue',
label='No Momentum, Interval of 30')

rects2 = plt.bar(index + bar_width, bpMomentum30, bar_width,
alpha=opacity,
color='wheat',
label='Momentum of 0.9, Interval of 30')

rects3 = plt.bar(index + bar_width*2, bpMomentum15, bar_width,
alpha=opacity,
color='red',
label='Momentum of 0.9, Interval of 15')

plt.ylim(0,1.0)
plt.xlabel('Number of Hidden Nodes', fontsize=font_size_large)
plt.ylabel('Classification Accuracy', fontsize=font_size_large)
plt.title('Side-by-Side dataset using MLP with Backpropagation\n', fontweight='bold', fontsize=font_size_large)
plt.xticks(index + bar_width, ('2', '4', '8', '16', '32', '64'), fontsize=font_size_large)
plt.yticks(fontsize=font_size_small)
plt.legend(fontsize=font_size_small, loc=4, shadow=True)


plt.tight_layout()
plt.show()
