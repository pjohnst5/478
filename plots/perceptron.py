import numpy as np
import matplotlib.pyplot as plt

n_groups = 3

accuracies = (0.5174242424, 0.4590909091, 0.8159090909)

fig, ax = plt.subplots()
index = np.arange(n_groups)
bar_width = 0.15
opacity = 0.8
font_size_large = 16
font_size_small = 13

rects1 = plt.bar(index + bar_width, accuracies, bar_width,
alpha=opacity,
color='orangered')

plt.ylim(0,1.0)
plt.xlabel('Data set', fontsize=font_size_large)
plt.ylabel('Classification Accuracy', fontsize=font_size_large)
plt.title('Perceptron accuracy with different data sets', fontweight='bold', fontsize=font_size_small)
plt.xticks(index + bar_width, ('Differences', 'Ratios', 'Side-by-side'), fontsize=font_size_large)
plt.yticks(fontsize=font_size_small)


plt.tight_layout()
plt.show()
