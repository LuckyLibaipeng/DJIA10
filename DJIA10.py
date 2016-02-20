# -*- coding: utf-8 -*-
"""
Created on Sat Feb 20 13:34:37 2016

@author: Administrator
"""
import numpy as np
import pandas as pd 
import pandas.io.data as web
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.decomposition import KernelPCA
symbols=['HD','NKE','AXP','JNJ','IBM','CSCO','KO','UNH','WMT','DJIA']
data=pd.DataFrame()
for sym in symbols:
    data[sym]=web.DataReader(sym,data_source='yahoo')['Close']
    data=data.dropna() 
dax=pd.DataFrame(data.pop('DJIA'))
data[data.columns[:6]].head()
scale_function=lambda x:(x-x.mean())/x.std()
#考虑多个成分的pca
pca=KernelPCA().fit(data.apply(scale_function))
len(pca.lambdas_)
pca.lambdas_[:10].round() 
#规范化
get_we=lambda x:x/x.sum()
get_we(pca.lambdas_)[:10]
get_we(pca.lambdas_)[:5].sum()
#构造pca指数
#只包含第一个成分的pca指数
pca=KernelPCA(n_components=1).fit(data.apply(scale_function))
dax['PCA_1']=pca.transform(-data)
dax.apply(scale_function).plot(figsize=(8,4))
#计算单个结果成分的加权平均数
pca=KernelPCA(n_components=5).fit(data.apply(scale_function))
pca_components=pca.transform(-data)
weights=get_we(pca.lambdas_)
dax['PCA_5']=np.dot(pca_components,weights)
dax.apply(scale_function).plot(figsize=(8,4))
#散点图
import matplotlib as mpl
mpl_dates=mpl.dates.date2num(data.index)
mpl_dates
plt.figure(figsize=(8,4))
plt.scatter(dax['PCA_5'],dax['DJIA'],c=mpl_dates)
lin_reg=np.polyval(np.polyfit(dax['PCA_5'],dax['DJIA'],1),dax['PCA_5'])
plt.plot(dax['PCA_5'],lin_reg,'r',lw=3)
plt.grid(True)
plt.xlabel('PCA_5')
plt.ylabel('DJIA')
plt.colorbar(ticks=mpl.dates.DayLocator(interval=250),format=mpl.dates.DateFormatter('%d %b %y'))

