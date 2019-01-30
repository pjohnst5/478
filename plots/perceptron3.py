import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots()

epochs1 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
RMSE1 = [0.3296902366978935, 0.273009453115974, 0.2672612419124244, 0.24922239313961342, 0.2553769592276246, 0.24291194120529402, 0.21583292365085782, 0.2553769592276246, 0.2553769592276246, 0.2613866510869666, 0.21583292365085782, 0.2613866510869666, 0.21583292365085782, 0.2672612419124244, 0.22291128503014113, 0.22291128503014113, 0.24291194120529402, 0.22291128503014113, 0.22977169333035918, 0.24291194120529402]

epochs2 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48]
RMSE2 = [0.3201319603689851, 0.2672612419124244, 0.2672612419124244, 0.2553769592276246, 0.21583292365085782, 0.22291128503014113, 0.20851441405707477, 0.2553769592276246, 0.23643312187173018, 0.23643312187173018, 0.21583292365085782, 0.24922239313961342, 0.20092951701393555, 0.18482827349523615, 0.22977169333035918, 0.24291194120529402, 0.21583292365085782, 0.22291128503014113, 0.22977169333035918, 0.19304683562633607, 0.20092951701393555, 0.21583292365085782, 0.17622684421256035, 0.22977169333035918, 0.22291128503014113, 0.18482827349523615, 0.17622684421256035, 0.21583292365085782, 0.23643312187173018, 0.17622684421256035, 0.19304683562633607, 0.15762208124782012, 0.17622684421256035, 0.21583292365085782, 0.18482827349523615, 0.22977169333035918, 0.18482827349523615, 0.23643312187173018, 0.17622684421256035, 0.19304683562633607, 0.20851441405707477, 0.20092951701393555, 0.19304683562633607, 0.22977169333035918, 0.20851441405707477, 0.22291128503014113, 0.20092951701393555, 0.17622684421256035, 0.18482827349523615]

epochs3 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58]
RMSE3 = [0.34802013220841155, 0.21583292365085782, 0.2786391062876764, 0.273009453115974, 0.22291128503014113, 0.20092951701393555, 0.22977169333035918, 0.18482827349523615, 0.22291128503014113, 0.28415724804218323, 0.24291194120529402, 0.22977169333035918, 0.23643312187173018, 0.20851441405707477, 0.18482827349523615, 0.24922239313961342, 0.21583292365085782, 0.20851441405707477, 0.18482827349523615, 0.21583292365085782, 0.23643312187173018, 0.19304683562633607, 0.17622684421256035, 0.22977169333035918, 0.21583292365085782, 0.22977169333035918, 0.21583292365085782, 0.24291194120529402, 0.20851441405707477, 0.24922239313961342, 0.22291128503014113, 0.20092951701393555, 0.20092951701393555, 0.23643312187173018, 0.22291128503014113, 0.17622684421256035, 0.20851441405707477, 0.23643312187173018, 0.21583292365085782, 0.20092951701393555, 0.23643312187173018, 0.20092951701393555, 0.22291128503014113, 0.19304683562633607, 0.23643312187173018, 0.22291128503014113, 0.20851441405707477, 0.20851441405707477, 0.21583292365085782, 0.15762208124782012, 0.20092951701393555, 0.22977169333035918, 0.19304683562633607, 0.22977169333035918, 0.22977169333035918, 0.21583292365085782, 0.22977169333035918, 0.21583292365085782, 0.22291128503014113]

epochs4 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71]
RMSE4 = [0.3611575592573076, 0.3102793772158663, 0.2613866510869666, 0.2672612419124244, 0.30010350181436113, 0.273009453115974, 0.22291128503014113, 0.2672612419124244, 0.2613866510869666, 0.24922239313961342, 0.20851441405707477, 0.2553769592276246, 0.2553769592276246, 0.2613866510869666, 0.2613866510869666, 0.19304683562633607, 0.23643312187173018, 0.24922239313961342, 0.273009453115974, 0.24291194120529402, 0.2672612419124244, 0.24291194120529402, 0.23643312187173018, 0.21583292365085782, 0.24291194120529402, 0.28415724804218323, 0.23643312187173018, 0.23643312187173018, 0.2613866510869666, 0.24291194120529402, 0.2613866510869666, 0.22291128503014113, 0.23643312187173018, 0.22291128503014113, 0.2672612419124244, 0.20851441405707477, 0.22977169333035918, 0.20851441405707477, 0.2672612419124244, 0.22291128503014113, 0.22977169333035918, 0.24291194120529402, 0.23643312187173018, 0.2613866510869666, 0.22977169333035918, 0.24922239313961342, 0.22977169333035918, 0.22291128503014113, 0.2672612419124244, 0.24291194120529402, 0.22977169333035918, 0.24922239313961342, 0.22977169333035918, 0.2613866510869666, 0.23643312187173018, 0.2672612419124244, 0.21583292365085782, 0.2613866510869666, 0.22977169333035918, 0.24291194120529402, 0.23643312187173018, 0.20092951701393555, 0.2613866510869666, 0.21583292365085782, 0.24291194120529402, 0.2613866510869666, 0.22977169333035918, 0.23643312187173018, 0.23643312187173018, 0.24922239313961342, 0.2553769592276246, 0.24291194120529402]

epochs5 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
RMSE5 = [0.3779644730092272, 0.22977169333035918, 0.28415724804218323, 0.2553769592276246, 0.2672612419124244, 0.24922239313961342, 0.2553769592276246, 0.24922239313961342, 0.21583292365085782, 0.22977169333035918, 0.22291128503014113, 0.23643312187173018, 0.22291128503014113, 0.22977169333035918]

maxLength = 72

diff1 = maxLength - len(epochs1)
diff2 = maxLength - len(epochs2)
diff3 = maxLength - len(epochs3)
diff5 = maxLength - len(epochs5)

for i in range(diff1):
    RMSE1.append(0.0)
for i in range(diff2):
    RMSE2.append(0.0)
for i in range(diff3):
    RMSE3.append(0.0)
for i in range(diff5):
    RMSE5.append(0.0)

avg = (np.array(RMSE1) + np.array(RMSE2) + np.array(RMSE3) + np.array(RMSE4) + np.array(RMSE5)) / 5

ax.plot(epochs4, avg)

ax.set(xlabel='Epochs', ylabel='Avg Misclassification Rate',
       title='Avg Misclassification Rate across Epochs')

ax.grid()

plt.show()
