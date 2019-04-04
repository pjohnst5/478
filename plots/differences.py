import numpy as np
import matplotlib.pyplot as plt

n_groups = 6

bpNoMomentum = (0.3954545455, 0.368939393939393, 0.4825757576, 0.5515151515, 0.4931818182, 0.4303030303)
bpMomentum30 =  (0.5545454545, 0.4886363636, 0.5053030303, 0.5689393939, 0.578030303, 0.4916666667)
bpMomentum15 = (0.5136363636, 0.4462121212, 0.471969697, 0.5189393939, 0.4651515152, 0.4825757576)

fig, ax = plt.subplots()
index = np.arange(n_groups)
bar_width = 0.15
opacity = 0.8
font_size_large = 16
font_size_small = 13

rects1 = plt.bar(index, bpNoMomentum, bar_width,
alpha=opacity,
color='b',
label='No Momentum, Interval of 30')

rects2 = plt.bar(index + bar_width, bpMomentum30, bar_width,
alpha=opacity,
color='g',
label='Momentum of 0.9, Interval of 30')

rects3 = plt.bar(index + bar_width*2, bpMomentum15, bar_width,
alpha=opacity,
color='r',
label='Momentum of 0.9, Interval of 15')

plt.ylim(0,1.0)
plt.xlabel('Number of Hidden Nodes', fontsize=font_size_large)
plt.ylabel('Classification Accuracy', fontsize=font_size_large)
plt.title('Differences dataset using MLP with Backpropagation\n', fontweight='bold', fontsize=font_size_small)
plt.xticks(index + bar_width, ('2', '4', '8', '16', '32', '64'), fontsize=font_size_large)
plt.yticks(fontsize=font_size_small)
plt.legend(fontsize=font_size_small)


plt.tight_layout()
plt.show()
