import numpy as np
import matplotlib.pyplot as plt

n_groups = 6
differencesAccuracies = (0.5147, 0.5658, 0.5521, 0.5446, 0.5113, 0.5258)
sidebysideAccuracies = (0.8432, 0.8416, 0.8304, 0.8234, 0.753, 0.8426)

fig, ax = plt.subplots()
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
plt.xlabel('Weka Model', fontsize=font_size_large)
plt.ylabel('Classification Accuracy', fontsize=font_size_large)
plt.title('Weka Model Accuracies (cont.)\n', fontweight='bold', fontsize=font_size_large)
plt.xticks(index + bar_width, ('JRip', 'Decision\nTable', 'PART', 'Simple\nLogistic', 'Lazy\nK\nStar', 'SGD'), fontsize=font_size_large)
plt.yticks(fontsize=font_size_small)
plt.legend(fontsize=font_size_small, loc=4, shadow=True)


plt.tight_layout()
plt.show()
