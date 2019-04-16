import numpy as np
import matplotlib.pyplot as plt

n_groups = 6
differencesAccuracies = (0.4727, 0.5746, 0.5519, 0.5542, 0.5273, 0.572)
sidebysideAccuracies = (0.4727, 0.8683, 0.8003, 0.8714, 0.8538, 0.8685)

index = np.arange(n_groups)
bar_width = 0.25
opacity = 0.8
font_size_large = 16
font_size_small = 13

rects1 = plt.bar(index + bar_width, differencesAccuracies, bar_width,
alpha=opacity,
color='turquoise',
label='Differences Dataset')

rects2 = plt.bar(index + bar_width*2, sidebysideAccuracies, bar_width,
alpha=opacity,
color='orange',
label='Side-by-Side Dataset')


plt.ylim(0,1.0)

plt.ylabel('Classification Accuracy', fontsize=font_size_large)
plt.title('Weka Model Accuracies\n', fontweight='bold', fontsize=font_size_large)
plt.xticks(index + bar_width, ('Baseline', 'OneR', 'J48', 'Random\nForest', 'Bayes\nNet', 'Naive\nBayes'), fontsize=font_size_large)
plt.yticks(fontsize=font_size_small)
plt.legend(fontsize=font_size_small, loc=4, shadow=True)


plt.tight_layout()
plt.show()
