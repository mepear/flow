import pandas as pd

tmp = pd.read_csv('./data/plot_0.csv')
name = tmp.columns
df = pd.DataFrame({'reward': []})
for i in name:
    if i != 'reward':
        df[i] = []

for i in range(100):
    tmp = pd.read_csv('./data/plot_{}.csv'.format(i))
    data = tmp.loc[0]
    df.loc[i] = data

df.to_csv("plot_reposition.csv", index=False, sep=',')