# -*- coding: utf-8 -*-

import pandas as pd
import random
import numpy as np
from sklearn import svm
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from time import time


class New_BDE():
    def __init__(self, NP, generation, data_X, data_y, feature_names):
        self.NP = NP  # Number of individuals in the population
        self.generation = generation
        self.data_X = data_X
        self.data_y = data_y
        self.feature_names = feature_names
        self.chrom_length = len(feature_names)
        self.F = 0   # Initialize the zoom factor
        self.CR = 1  # Initialize the cross factor
        self.max_x = []  # Individuals with the greatest fitness
        self.max_f = []  # Maximum fitness

    # Initial population
    def initialtion(self):
        pop = [[]]
        for i in range(self.NP):
            tempt = []
            for j in range(self.chrom_length):
                if random.random() >= 0.85:
                    tempt.append(1)
                else:
                    tempt.append(0)
            pop.append(tempt)
        return pop[1:]


    def tanh(self, x):
        return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))


    def cal_CR(self, gv):
        cr = 0.9 * (1 - self.tanh(gv))
        return cr
    
    # mutation operator

    def mutation(self, np_list):
        v_list = []
        for i in range(0, self.NP):
            v_tempt = []
            r1 = random.randint(0, self.NP-1)
            while r1 == i:
                r1 = random.randint(0, self.NP-1)
            r2 = random.randint(0, self.NP-1)
            while r2 == r1 | r2 == i:
                r2 = random.randint(0, self.NP-1)
            r3 = random.randint(0, self.NP-1)
            while r3 == r2 | r3 == r1 | r3 == i:
                r3 = random.randint(0, self.NP-1)
            for j in range(0, self.chrom_length):
                if np_list[r1][j] == np_list[r2][j]:
                    tempt = 0
                else:
                    tempt = self.F*np_list[r1][j]
                pr = self.tanh(tempt)
                if pr > random.random() and np_list[r3][j] == 1:
                    v_tempt.append(1)
                else:
                    v_tempt.append(0)
            v_list.append(v_tempt)
        return v_list
    
    # Crossover operator

    def crossover(self, np_list, v_list):
        u_list = []
        for i in range(0, self.NP):
            vv_list = []
            for j in range(0, self.chrom_length):
                if (random.random() <= self.CR) | (j == random.randint(0, self.chrom_length-1)):
                    vv_list.append(v_list[i][j])
                else:
                    vv_list.append(np_list[i][j])
            u_list.append(vv_list)
        return u_list
    
    # Calculate fitness

    def calFitness(self, C_pop):

        obj_value = []
        for i in range(len(C_pop)):
            data_X_test = self.data_X
            for j in range(len(C_pop[i])):
                if C_pop[i][j] == 0:
                    data_X_test = data_X_test.drop(
                        self.feature_names[j+1], axis=0)
            clf = svm.SVC()
            score = cross_val_score(
                clf, data_X_test.T, self.data_y, cv=5, scoring='accuracy').mean()

            obj_value.append(score)
        return obj_value
    
    # selection

    def selection(self, u_pop, pop):
        obj = []  
        obj_u = [] 
        obj = self.calFitness(pop)
        obj_u = self.calFitness(u_pop)
        for i in range(0, self.NP):
            if obj_u[i] >= obj[i]:
                pop[i] = u_pop[i]
            else:
                pop[i] = pop[i]
        return pop

    def output(self):
        max_ff = max(self.max_f)
        max_xx = self.max_x[self.max_f.index(max_ff)]

        print('The optimal individual： ')
        print(max_xx)
        print('Optimal individual length：')
        print(max_xx.count(1))
        print('The optimal individual fitness：')
        print(max_ff)

        x_label = np.arange(0, self.generation+1, 1)
        plt.plot(x_label, self.max_f, color='red')
        plt.xlabel('iteration')
        plt.ylabel('best f(x)')
        plt.savefig('./iteration-picture.png')
        plt.show()
        
   

    def BDE(self):
        pop = self.initialtion()

        obj_value = []
        obj_value = self.calFitness(pop)
        self.max_f.append(max(obj_value))
        self.max_x.append(pop[obj_value.index(max(obj_value))])
        
        # Evolutionary process
        for i in range(1, self.generation+1):
            print("Current iteration number %d！" % i)
            if (i/self.generation) < 0.5:
                self.F = random.uniform(0.5, 1)
            else:
                self.F = random.uniform(0, 0.5)
            v_pop = self.mutation(pop)
            u_pop = self.crossover(pop, v_pop)
            pop = self.selection(u_pop, pop)
            obj_value = []
            obj_value = self.calFitness(pop)
            self.CR = self.cal_CR(i/self.generation)
            self.max_f.append(max(obj_value))
            self.max_x.append(pop[obj_value.index(max(obj_value))])

        self.output()


# Read data
def readfile(fname):
    df = pd.read_excel(fname, header=None)
    data = pd.DataFrame(df)
    feature_names = data.iloc[1::, 0] 
    X = data.iloc[1::, 1::]
    y = data.iloc[0, 1::].values
    y = np.array(y, dtype='int32')
    X.set_index(feature_names, inplace=True)
    return X, y, feature_names


if __name__ == "__main__":
    start = time()
    X, labels, feature_names = readfile("./Filter_Result/Filtered_data.xlsx")
    data_X = X.iloc[0:, :]
    data_y = labels
    feature_names = feature_names[0:]
    NP = 20
    generation = 500
    new_bde = New_BDE(NP, generation, data_X, data_y, feature_names)
    new_bde.BDE()
    end = time()
    print('Running time: %s Second' % (end - start))


