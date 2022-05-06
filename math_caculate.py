import constants as c

import matplotlib.pyplot as plt

input_txt = c.PRE_CHECKPOINT_FOLDER + "/train_loss_acc.txt"
x = []
y = []

f = open(input_txt)

for line in f:
    line = line.strip('\n')
    line = line.split(',')

    x.append(str(line[0]))
    y.append(float(line[2]))

f.close

plt.plot(x, y, marker='o', label='lost plot')
plt.xticks(x[0:len(x):2], x[0:len(x):2], rotation=45)
plt.margins(0)
plt.xlabel("train step")
plt.ylabel("lost")
plt.title("matplotlip plot")
plt.tick_params(axis="both")

plt.show()