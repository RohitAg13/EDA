import numpy as np
import pandas as pd
import math

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from sklearn.utils.extmath import cartesian

sns.set()




class TreeEnsemble():
    def __init__(self, x, y, n_trees, sample_sz, min_leaf=5):
        self.x,self.y,self.sample_sz,self.min_leaf = x,y,sample_sz,min_leaf
        self.trees = [self.create_tree() for i in range(n_trees)]

    def create_tree(self):
        idxs = np.random.permutation(len(self.y))[:self.sample_sz]
        return DecisionTree(self.x.iloc[idxs], self.y[idxs], 
                    idxs=np.array(range(self.sample_sz)), min_leaf=self.min_leaf)
        
    def predict(self, x):
        return np.mean([t.predict(x) for t in self.trees], axis=0)




class DecisionTree():
    def __init__(self, x, y, idxs, min_leaf=5):
        self.x,self.y,self.idxs,self.min_leaf = x,y,idxs,min_leaf
        self.n,self.c = len(idxs), x.shape[1]
        self.val = np.mean(y[idxs])
        self.score = float('inf')
        self.find_varsplit()
        
    def find_varsplit(self):
        for i in range(self.c): self.find_better_split(i)
        if self.score == float('inf'): return
        x = self.split_col
        lhs = np.nonzero(x<=self.split)[0]
        rhs = np.nonzero(x>self.split)[0]
        self.lhs = DecisionTree(self.x, self.y, self.idxs[lhs], self.min_leaf)
        self.rhs = DecisionTree(self.x, self.y, self.idxs[rhs], self.min_leaf)

    def find_better_split(self, var_idx):
        x,y = self.x.values[self.idxs,var_idx], self.y[self.idxs]
        sort_idx = np.argsort(x)
        sort_y,sort_x = y[sort_idx], x[sort_idx]
        rhs_cnt,rhs_sum,rhs_sum2 = self.n, sort_y.sum(), (sort_y**2).sum()
        lhs_cnt,lhs_sum,lhs_sum2 = 0,0.,0.
        for i in range(self.min_leaf, self.n-self.min_leaf-1):
            xi,yi = sort_x[i],sort_y[i]
            lhs_cnt += 1; rhs_cnt -= 1
            lhs_sum += yi; rhs_sum -= yi
            lhs_sum2 += yi**2; rhs_sum2 -= yi**2
            if i<self.min_leaf or xi==sort_x[i+1]:
                continue

            lhs_std = std_aggr(lhs_cnt, lhs_sum, lhs_sum2)
            rhs_std = std_aggr(rhs_cnt, rhs_sum, rhs_sum2)
            curr_score = lhs_std*lhs_cnt + rhs_std*rhs_cnt
            if curr_score<self.score:
                self.var_idx,self.score,self.split = var_idx,curr_score,xi

    @property
    def split_name(self): return self.x.columns[self.var_idx]
    
    @property
    def split_col(self): return self.x.values[self.idxs,self.var_idx]

    @property
    def is_leaf(self): return self.score == float('inf')
    
    def __repr__(self):
        s = f'n: {self.n}; val:{self.val}'
        if not self.is_leaf:
            s += f'; score:{self.score}; split:{self.split}; var:{self.split_name}'
        return s

    def predict(self, x):
        return np.array([self.predict_row(xi) for xi in x])

    def predict_row(self, xi):
        if self.is_leaf: return self.val
        t = self.lhs if xi[self.var_idx]<=self.split else self.rhs
        return t.predict_row(xi)
def std_aggr(cnt, s1, s2): 
	return math.sqrt((s2/cnt) - math.pow((s1/cnt),2))

def plot_splits(tree, wll, wur, depth = 1, max_depth = 300, show_val = False):
    if max_depth < depth or tree.is_leaf:
        if show_val:
            plt.text((wll[0] + wur[0]) / 2 - (wur[0] - wll[0]) / 2.2, (wll[1] + wur[1]) / 2, 
                     f'value: {round(tree.val, 2)}\n# in partition: {len(tree.idxs)}')
        return
    
    feature = tree.split_name
    thresh = tree.split
    
    if feature == 'x':
        plt.vlines([thresh], wll[1], wur[1])
        plot_splits(tree.lhs, wll, (thresh, wur[1]), depth + 1, max_depth, show_val)
        plot_splits(tree.rhs, (thresh, wll[1]), wur, depth + 1, max_depth, show_val)
        
    else:
        plt.hlines([thresh], wll[0], wur[0])
        plot_splits(tree.lhs, wll, (wur[0], thresh), depth + 1, max_depth, show_val)
        plot_splits(tree.rhs, (wll[0], thresh) , wur, depth + 1, max_depth, show_val)