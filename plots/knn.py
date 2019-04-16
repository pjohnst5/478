import numpy as np
import matplotlib.pyplot as plt

n_groups = 6

differences = (0.4992424242, 0.4825757576, 0.4863636364, 0.4977272727, 0.5007575758, 0.4977272727)
ratios =  (0.4734848485, 0.5159090909, 0.5212121212, 0.5439393939, 0.6106060606, 0.5416666667)
sidebyside = (0.7439393939, 0.8409090909, 0.8424242424, 0.8515151515, 0.8227272727, 0.8151515152)
ninetyten = (0.6363636364, 0.3636363636, 0.3636363636, 0.2727272727, 0.3636363636, 0.3636363636)

fig, ax = plt.subplots()
index = np.arange(n_groups)
bar_width = 0.15
opacity = 0.8
font_size_large = 16
font_size_small = 13

rects1 = plt.bar(index, differences, bar_width,
alpha=opacity,
color='#3369E8',
label='Differences Dataset')

rects2 = plt.bar(index + bar_width, ratios, bar_width,
alpha=opacity,
color='#D50F25',
label='Ratios Dataset')

rects3 = plt.bar(index + bar_width*2, sidebyside, bar_width,
alpha=opacity,
color='#EEB211',
label='Side-by-Side Dataset')

rects3 = plt.bar(index + bar_width*3, ninetyten, bar_width,
alpha=opacity,
color='#009925',
label='Individual Teams Dataset')


plt.ylim(0,1.0)
plt.xlabel('Number of Neighbors', fontsize=font_size_large)
plt.ylabel('Classification Accuracy', fontsize=font_size_large)
plt.title('KNN Accuracy on Datasets', fontweight='bold', fontsize=font_size_small)
plt.xticks(index + bar_width, ('1', '3', '5', '7', '9', '11'), fontsize=font_size_large)
plt.yticks(fontsize=font_size_small)
plt.legend(fontsize=font_size_small, loc=3, shadow=True)


plt.tight_layout()
plt.show()
