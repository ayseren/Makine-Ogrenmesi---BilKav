"""
Created on Thu Mar 15 04:18:20 2018

@author: ayseguleren
"""

#1. kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#2. Veri Onisleme
#2.1. Veri Yukleme
veriler = pd.read_csv('odev_tenis.csv')

#veri on isleme
#encoder:  Kategorik -> Numeric
#hepsini ayni anda encoding yapabiliriz:
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
verilerEnc = veriler.apply(le.fit_transform)

outlook = verilerEnc.iloc[:,:1]
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
ct = ColumnTransformer([("Name_Of_Your_Step", OneHotEncoder(),[0])], remainder="passthrough")
outlook = ct.fit_transform(outlook)

havadurumu = pd.DataFrame(data = outlook, index = range(14), columns=['o','r','s'])
#temperature ve humidity
sonveriler = pd.concat([havadurumu,veriler.iloc[:,1:3]],axis = 1)
#windy ve play
sonveriler = pd.concat([verilerEnc.iloc[:,-2:],sonveriler], axis = 1)
print(sonveriler)

#verilerin egitim ve test icin bolunmesi
from sklearn.model_selection import train_test_split
x_train, x_test,y_train,y_test = train_test_split(sonveriler.iloc[:,:-1],sonveriler.iloc[:,-1:],test_size=0.33, random_state=0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)

y_pred = regressor.predict(x_test)

print(y_pred)

import statsmodels.formula.api as sm 
X = np.append(arr = np.ones((14,1)).astype(int), values=sonveriler.iloc[:,:-1], axis=1 )
X_l = sonveriler.iloc[:,[0,1,2,3,4,5]].values
r_ols = sm.OLS(endog = sonveriler.iloc[:,-1:], exog =X_l)
r = r_ols.fit()
print(r.summary())

sonveriler = sonveriler.iloc[:,1:]

import statsmodels.formula.api as sm 
X = np.append(arr = np.ones((14,1)).astype(int), values=sonveriler.iloc[:,:-1], axis=1 )
X_l = sonveriler.iloc[:,[0,1,2,3,4]].values
r_ols = sm.OLS(endog = sonveriler.iloc[:,-1:], exog =X_l)
r = r_ols.fit()
print(r.summary())

x_train = x_train.iloc[:,1:]
x_test = x_test.iloc[:,1:]

regressor.fit(x_train,y_train)


y_pred = regressor.predict(x_test)







