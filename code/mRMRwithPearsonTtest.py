# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import warnings
from scipy.stats import ttest_ind
from scipy.stats import pearsonr
import copy
import os

class mRMR():

    def __init__(self, feature_num):
        self.feature_num = feature_num
        self._selected_features = []
        self.accs = []
        self._selected_tvalues = []

    def entropy(self, c):
        c_normalized = c/float(np.sum(c))
        c_normalized = c_normalized[np.nonzero(c_normalized)]
        H = - sum(c_normalized*np.log2(c_normalized))
        return H

    def feature_label_Ttests(self, arr, y):
        '''
        calculate feature-label mutual information
        '''
        [raw, col] = arr.shape
        X = arr
        print(y)

        pos = np.where(y == 1)[0]
        neg = np.where(y == 0)[0]

        tValues = []
        pValues = []
        for j in range(col):
            tempT, tempP = ttest_ind(X[pos, j].astype(np.float), X[neg, j].astype(np.float))
            if np.isnan(tempT):
                tempT = 0
                tempP = 1
            tValues.append(abs(tempT))
            pValues.append(abs(tempP))
        tValues = np.array(tValues)
        print(tValues.shape)
        tValues = tValues/tValues.max()
        return tValues

    def feature_feature_Pearsons(self, x1, x2):
        '''
        calculate feature-faeature mutual information
        '''
        pear = pearsonr(x1, x2)
        if np.isnan(pear)[0]:
            pear = [0,1]
        return np.abs(pear[0])

    def fit(self, X, y):
        if self.feature_num > X.shape[1]:
            self.feature_num = X.shape[1]
            warnings.warn("feature_num should be less than or equal to the number of features %d" % X.shape[1], UserWarning)

        tValues = self.feature_label_Ttests(X, y)
        self.tValues_origin = copy.deepcopy(tValues)
        max_tValue_arg = np.argmax(tValues) 
        print("max_tValue_arg: ",np.argmax(tValues))

        selected_features = []

        tValues = list(zip(range(len(tValues)),tValues))
        print("tValues",tValues)
        temp = tValues.pop(int(max_tValue_arg))
        selected_features.append(temp)

        while True:
            max_theta = float("-inf")
            max_theta_index = None

            for tvalue_outset in tValues:
                ff_tvalues = []
                for tvalue_inset in selected_features:
                    ff_tvalue = self.feature_feature_Pearsons(X[:,tvalue_outset[0]], X[:,tvalue_inset[0]])
                    ff_tvalues.append(ff_tvalue)

                theta = tvalue_outset[1] - 1/len(selected_features) * sum(ff_tvalues)
                if theta >= max_theta:
                    max_theta = theta
                    max_theta_index = tvalue_outset
            selected_features.append(max_theta_index)
            print("max_theta_index:", max_theta_index)
            tValues.remove(max_theta_index)
            print("The currently selected feature is：%s" % selected_features)

            if len(selected_features) >= self.feature_num:
                break
        print(self.accs)
        self._selected_features = [ind for ind,tvalue in selected_features]
        self._selected_tvalues = [tvalue for ind, tvalue in selected_features]
        return self
    
    def transform(self, X):
        return X[:, self._selected_features]

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)

    def important_features(self):
        return self._selected_features

def ttest(data):
    [raw, col] = data.shape
    X = data.iloc[1:, 1:].values
    y = data.iloc[0, 1:].values.astype('int')
    print(y)

    pos = np.where(y==1)[0]
    neg = np.where(y==0)[0]
    pValue = []
    for i in range(raw - 1):
        tempT, tempP = ttest_ind(X[i, pos].astype(np.float), X[i, neg].astype(np.float))
        pValue.append(tempP)
    pValue = np.array(pValue)
    selectedNo = np.array(np.where(pValue < 0.05))[0] + 1
    selectedX = data.iloc[selectedNo, :]
    column = pd.DataFrame(data.iloc[0, :].values.reshape(1,-1))
    selected_data = pd.concat([column,selectedX],axis=0)
    print("Before screening：%s\t After screening：%s" % (data.shape, selected_data.shape))
    print("After screening：",selected_data)
    return selected_data

def getInfo(data):
    X = data.iloc[1:, 1:].values.astype('float').T
    y = data.iloc[0, 1:].values.astype('int')
    print("y shape:",y.shape)
    feature_Name = data.iloc[1:, 0].values
    pos_X = np.nan_to_num(np.log2(X+1e-5))
    neg_X = -np.nan_to_num(np.log2(-X+1e-5))
    X = pos_X + neg_X
    return X,y,feature_Name

if __name__ == '__main__':
    for i in [0]:
        filePath = r".\Datasets"
        fileList = os.listdir(filePath)
        for file in fileList:
            fileName = file.split('.')[0]
            fileType = file.split('.')[-1]
            print(file)
            if fileType == 'xlsx':
                data = pd.read_excel(filePath+'\\'+file, header=None)
                X, y, feature_Name = getInfo(data)
                print("y:", y)

                mrmr = mRMR(500)
                x_ = mrmr.fit_transform(X, y)

                df = list(zip(mrmr._selected_features, feature_Name[np.array(mrmr._selected_features)], mrmr._selected_tvalues))
                df = pd.DataFrame(df)
                df.columns = ['Feature number', 'Feature name', 'Mutual information']
                print(df)

                acc_df = pd.DataFrame(mrmr.accs)
                print(acc_df)

                writer = pd.ExcelWriter(r'.\Filter_Datasets\%s_mRMR.xlsx' % (fileName))
                df.to_excel(writer, sheet_name='Filtered characteristics', index=0)
                writer.save()