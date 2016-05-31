import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import numpy as np
from scipy.interpolate import interp1d
import sys


def load():
    return pd.read_excel("Analysis.xls", header=None)


data = load()
for i in range(17):
    print(data[i].dropna().describe())

for i in range(3, 11):
    plt.hist(data[i].dropna())
    plt.show()
    sm.qqplot(data[i], line='q')
    plt.show()
    plt.boxplot(data[i].dropna())
    plt.show()

for i in range(11, 18):
    h = data[data[2].isin(['high'])][i]
    m = data[data[2].isin(['medium'])][i]
    l = data[data[2].isin(['low'])][i]
    d = [np.asarray(h), np.asarray(m), np.asarray(l)]
    plt.boxplot(d)
    plt.show()


def getmaxcorr(dt, index):
    max = -1.0
    pos = 0;
    for i in range(3, 11):
        if (dt.corr(data[i]) > max and i != index):
            max = dt.corr(data[i])
            pos = i
    return pos


def dist(a, b):
    return np.sum((a - b) * (a - b))


def getnn(index):
    min = sys.maxint
    pos = 0
    for i in range(0, 199):
        if (i != index):
            d = []
            for j in range(3, 11):
                if (np.isnan(data.loc[index, j]) or np.isnan(data.loc[i, j])):
                    continue
                else:
                    d.append(j)
            t = data.loc[[index, i], d]
            a = np.asarray(t.loc[index])
            b = np.asarray(t.loc[i])
            if (dist(a, b) < min):
                min = dist(a, b)
                pos = i
    return pos


def getnullpos(t):
    return np.asarray(t.index[t.isnull()], dtype=np.int)


def delnull():
    data1 = data.copy()
    data1 = data1.dropna()
    print
    len(data1)
    data1.to_excel('delnull.xls')


def getfrec_num(plist):
    dic = {}
    for i in plist:
        if (dic.get(i) == None):
            dic[i] = 1
        else:
            dic[i] = dic[i] + 1
    list = sorted(dic.iteritems(), key=lambda d: d[1], reverse=True)
    print
    list[0][0]
    return list[0][0]


def fillwithfreq():
    data2 = data.copy()
    for i in range(3, 11):
        print
        i
        data2[i] = data2[i].fillna(getfrec_num(np.asarray(data2[i].dropna())))
        if (i == 10):
            print
            data2[i]
    data2.to_excel('fillwithfreq.xls')


def fillwithattr():
    data3 = data.copy()
    for i in range(3, 11):
        corrvec = getmaxcorr(data3[i], i)
        table_2 = data3[[i, corrvec]].dropna()
        x = table_2[corrvec]
        y = table_2[i]
        f = interp1d(x, y, bounds_error=False)
        for j in getnullpos(data3[i]):
            if (np.isnan(data3[corrvec][j])):
                data3[i][j] = data3[corrvec].median()
            else:
                data3[i][j] = f(data3[corrvec][j])
    data3.to_excel('fillwithattr.xls')


def fillwithobj():
    data4 = data.copy()
    for i in range(0, 199):
        if (len(np.asarray(data4.loc[i][data4.loc[i].isnull()])) != 0):
            index = getnn(i)
            data4.loc[i] = data4.loc[i].fillna(data4.loc[index, 3:11].median())
    data4.to_excel('fillwithobj.xls')


delnull()
fillwithfreq()
fillwithattr()
fillwithobj()
