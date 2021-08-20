import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("plot.csv")
# for i in range(100):
#     line = np.array(df.loc[i])
#     reward = line[0:1]
#     other = line[1:]
#     other = other / other.max()
#     line = np.concatenate((reward, other))
#     df.loc[i] = line

dpi = 1000
fig, ax = plt.subplots()
index = ['left2_0_0', 'bot2_1_0', 'left2_1_0', 'top1_1_0']

for i in index:
    if i != 'reward':
        X = np.array(df['reward'])
        # X = X[1:] - X[:-1]
        # X = X[1:] - X[:-1]
        Y = np.array(df[i])
        # Y = Y[1:] - Y[:-1]
        # Y = Y[1:] - Y[:-1]
        plt.scatter(Y, X, s=64, marker='o', alpha=0.3, linewidths=1, edgecolors='w', label=i)
        # try:
        #     f = np.polyfit(Y, X, 1)
        #     p = np.poly1d(f)
        #     X_pre = p(Y)
        #     plt.plot(Y, X_pre)
        # except:
        #     a = 1

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

# plt.xlim(-1.5, 1.5)
# plt.ylim(-1.5, 1.5)
plt.xlabel('Flow')
plt.ylabel('Reward')
# plt.legend()

plt.savefig('./taxi1/origin/focus_origin', dpi=dpi)