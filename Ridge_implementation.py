# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 01:10:32 2019

@author: Prishat
"""

import numpy as np
import pandas as pd

epochs=1000
alpha=0.1
beta=0.5

with open('Train_f.csv') as fp:
    data=pd.read_csv(fp)

data.dropna(inplace=True)   
x=np.array(data.drop(["Loan_ID","11"],axis=1))
y=np.array(data["11"].values)
y=y.reshape(y.shape[0],1)

x_train=x[:400]
y_train=y[:400]
x_val=x[-80:]
y_val=y[-80:]

def _cost(x,y,w,beta):
    m=x.shape[0]
    C=(1./(2.*m))*np.sum((x.dot(w)-y)**2 + (beta*np.dot(w.T,w)))
    return C

def _standardize(x):
    m=x.mean()
    std=x.std()
    x=x-m
    x/=std
    #print("x=",x)
    return x
    
def _gradient_descent(x,y,w,alpha,beta):
    m=x.shape[0]
    C=_cost(x,y,w,beta)
    update=(alpha/m)*(np.dot(x.T,(x.dot(w)-y)) + beta*w)
    w-=update
    return w,C
    
def _fit_gd(x,y,alpha,beta,epochs):
    x=_standardize(x)
    y=_standardize(y)
    w=np.zeros([x.shape[1],1])
    loss=[]
    for i in range(epochs):
        loss.append(_gradient_descent(x,y,w,alpha,beta))
        #print(loss[i])
        w=loss[i][0]
        
    return w

def _predict(x,w):
    y_hat=x.dot(w)
    m=y_hat.mean()
    for i in range(len(y_hat)):
        if y_hat[i]>m:
            y_hat[i]=1
        else:
            y_hat[i]=0
    return y_hat

def _accuracy(y,y_hat):
    s=0
    t=y.shape[0]
    for i in range(t):
        if(y[i]==y_hat[i]):
            s+=1
    return float((s/t)*100)
    

w_1=_fit_gd(x_train,y_train,0.1,1,1000)
y_1=_predict(x_val,w_1)
acc_1=_accuracy(y_val,y_1)
print(acc_1,"%") 
   