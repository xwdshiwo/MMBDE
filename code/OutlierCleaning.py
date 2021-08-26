# -*- coding: utf-8 -*-

import os
import pandas as pd
from sklearn import preprocessing
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False



def outlierProcess(df):
    mean = df.mean()
    sigma = df.std()
    Max = mean + sigma*3
    Min = mean - sigma*3
    data_nor = df[(df >= Min) & (df <= Max)]
    data_nor = pd.DataFrame(data_nor)
    return data_nor

if __name__=='__main__':
    filePath = r".\DatasetsName"
    fileList = os.listdir(filePath)
    for file in fileList:
        fileName = file.split('.')[0]
        print(fileName)
        fileType = file.split('.')[-1]
        if fileType=='xlsx' and fileName in ['Colon', 'Lymphoma', 'Leukemia', 'Prostate']:
            data = pd.read_excel(filePath + '\\' + file, header=None)
            print(type(data), data)
            X = data.iloc[1:-1, 1:].values.astype('float').T
            X = pd.DataFrame(X)
            print(X)
            Y = data.iloc[0, 1:].values
            
            le = preprocessing.LabelEncoder()
            Y = le.fit_transform(Y)
            print("Y:", Y)

            df2 = X.copy()
            print(df2)
            data_nor = outlierProcess(df2)
            print(type(data_nor), data_nor.shape)
            print(data_nor.T)
            data.iloc[1:-1, 1:] = data_nor.T.values
            data.iloc[0,1:] = Y
            
            data[0:].to_excel(r'.\Outlier cleaning'+'\\'+fileName+'.xlsx', index=0, header=False)
