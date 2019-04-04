import numpy as np
import matplotlib.pyplot as plt

n_groups = 6

fiftyfifty = (0.5087719298, 0.4736842105, 0.5263157895, 0.5087719298, 0.5087719298, 0.5438596491)
seventythirty =  (0.5882352941, 0.3529411765, 0.3823529412, 0.4117647059, 0.5588235294, 0.5294117647)
ninetyten = (0.6363636364, 0.3636363636, 0.3636363636, 0.2727272727, 0.3636363636, 0.3636363636)

fig, ax = plt.subplots()
index = np.arange(n_groups)
bar_width = 0.15
opacity = 0.8
font_size_large = 16
font_size_small = 13

rects1 = plt.bar(index, fiftyfifty, bar_width,
alpha=opacity,
color='lime',
label='50/50 split train/test data')

rects2 = plt.bar(index + bar_width, seventythirty, bar_width,
alpha=opacity,
color='orange',
label='70/30 split train/test data')

rects3 = plt.bar(index + bar_width*2, ninetyten, bar_width,
alpha=opacity,
color='deeppink',
label='90/10 split train/test data')

plt.ylim(0,1.0)
plt.xlabel('Number of Neighbors', fontsize=font_size_large)
plt.ylabel('Classification Accuracy', fontsize=font_size_large)
plt.title('Highest Regression Score Comparison using KNN\n', fontweight='bold', fontsize=font_size_small, loc='center')
plt.xticks(index + bar_width, ('1', '3', '5', '7', '9', '11'), fontsize=font_size_large)
plt.yticks(fontsize=font_size_small)
plt.legend(fontsize=font_size_small)


plt.tight_layout()
plt.show()
