import csv
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram
import matplotlib.pyplot as plt

def load_data(filepath):
    l = list()
    with open(filepath) as csvfile:
        reader = csv.DictReader(csvfile)
        key = ['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed']
        for row in reader:
            d = {k: row[k] for k in row.keys() if k in key}
            l.append(d)
    return l

def calc_features(row):
    x = [row['Attack'],row['Sp. Atk'],row['Speed'],row['Defense'],row['Sp. Def'],row['HP']]
    return np.array(x, dtype="int64")

def hac(features):
    n = len(features)
    Z = np.zeros(shape=(n-1, 4))
    numbers = list(range(n))
    dist = [np.linalg.norm(features[i]-features[j]) for j in range(n) for i in range(n)]
    df = pd.DataFrame(np.array(dist).reshape(n,n), columns = numbers)
    for i in range(n-1):
        d = df.to_numpy()
        d_upper = np.triu(d)
        dmin = np.min(d[np.triu_indices(df.shape[0], 1)])
        index1,index2 = np.where(d_upper==dmin)
        index1 = list(index1) 
        index2 = list(index2)
        names = df.columns
        x = names[index1[0]]
        y = names[index2[0]]
        Z[i,0] = x
        Z[i,1] = y
        Z[i,2] = dmin
        if x <= n-1 and y <= n-1:
            Z[i,3] = 2
        elif x <= n-1 and y > n-1:
            Z[i,3] = Z[y-n,3]+1
        elif x > n-1 and y > n-1:
            Z[i,3] = Z[y-n,3]+Z[x-n,3]
        x_values = df[x].to_numpy()
        y_values = df[y].to_numpy()
        new = list(np.maximum(x_values,y_values))
        df[n+i] = new
        new.append(0)
        df.loc[n+i] = new
        df = df.drop([x,y]).drop([x,y],axis=1)
    return Z

def imshow_hac(Z):
    dendrogram(Z)
    plt.show()