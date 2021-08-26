# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn import preprocessing

def getInfo(data):
    X = data.iloc[1:, 1:].values.astype('float').T
    X = pd.DataFrame(X)
    le = preprocessing.LabelEncoder()
    y = le.fit_transform(np.array(data.iloc[0, 1:]))
    feature_Name = data.iloc[1:, 0].values
    return X,y,feature_Name

def KNN_fill(X, feature_Name):
    # Set the number of neighbors to 3
    imputer = KNNImputer(n_neighbors=3)
    fill_X = imputer.fit_transform(X)
    print(feature_Name.shape, fill_X.shape)
    fill_X = np.concatenate((feature_Name.reshape((1,-1)), fill_X), axis=0)
    return fill_X

if __name__=='__main__':
    filePath = r".\DatasetsName"
    fileList = os.listdir(filePath)
    for file in fileList:
        fileName = file.split('_abnormal')[0]
        fileType = file.split('.')[1]
        print(file)
        if fileType=='xlsx' and fileName in ['Colon', 'Lymphoma', 'Leukemia', 'Prostate']:
            data = pd.read_excel(filePath + '\\' + file, header=None)
            X, y, feature_Name = getInfo(data)
            columns = data.iloc[0, 0:].values.reshape(1,-1)
            columns = pd.DataFrame(columns)

            print("X:" ,type(X))
            fill_X = KNN_fill(X, feature_Name)

            fillDf = pd.DataFrame(fill_X).T
            fillDf = pd.concat([columns,fillDf], axis=0)
            fillDf.to_excel(r'.\ResultsDatasets\%s_KNNFill.xlsx'% fileName, header=0, index=0 )