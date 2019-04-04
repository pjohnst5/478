import numpy as np
import matplotlib.pyplot as plt

n_groups = 6

differences = ()
ratios =  ()
sidebyside = ()

fig, ax = plt.subplots()
index = np.arange(n_groups)
bar_width = 0.15
opacity = 0.8
font_size_large = 16
font_size_small = 13

rects1 = plt.bar(index, differences, bar_width,
alpha=opacity,
color='indigo',
label='Differences Dataset')

rects2 = plt.bar(index + bar_width, bpMomentum30, bar_width,
alpha=opacity,
color='crimson',
label='Ratios Dataset')

rects3 = plt.bar(index + bar_width*2, bpMomentum15, bar_width,
alpha=opacity,
color='mediumturquoise',
label='Side-by-Side Dataset')

plt.ylim(0,1.0)
plt.xlabel('Number of Hidden Nodes', fontsize=font_size_large)
plt.ylabel('Classification Accuracy', fontsize=font_size_large)
plt.title('Ratios dataset using MLP with Backpropagation\n', fontweight='bold', fontsize=font_size_small)
plt.xticks(index + bar_width, ('2', '4', '8', '16', '32', '64'), fontsize=font_size_large)
plt.yticks(fontsize=font_size_small)
plt.legend(fontsize=font_size_small)


plt.tight_layout()
plt.show()
