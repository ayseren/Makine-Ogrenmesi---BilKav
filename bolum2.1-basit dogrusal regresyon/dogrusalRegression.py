# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 18:50:13 2020

@author: sadievrenseker
"""

#1.kutuphaneler
import matplotlib.pyplot as plt
import pandas as pd


#################################### TAHMIN ######################################################
"""
Kategorik veriler siniflandirilir; sayisal veriler tahmin edilir

Forecasting hic tahmin edilmemisleri tahmin etme;
prediction gecmis verileri o anki verilerle tahmin etme
"""

veriler = pd.read_csv('satislar.csv')

aylar = veriler[['Aylar']]
satislar = veriler[['Satislar']] #veriler.iloc[:,:1].values

#verilerin egitim ve test icin bolunmesi
from sklearn.model_selection import train_test_split
#from sklearn.cross_validation import train_test_split videoda yazilmis

x_train,x_test,y_train,y_test = train_test_split(aylar,satislar,test_size=0.33, random_state=0)

#verilerin olceklenmesi
from sklearn.preprocessing import StandardScaler

sc=StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)
Y_train = sc.fit_transform(y_train)
Y_test = sc.transform(y_test)

#linear regression modelini olusturma
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x_train,y_train)

y_pred = lr.predict(x_test)
#print(y_pred)
#print(y_test)

#gorsellestirme

#plt.plot(x_train, y_train) 
#hatali bir grafik verir cunku veriler sirali olarak gelmiyor
#verileri hazirlarken random_state vermemizle de alakali

x_train = x_train.sort_index() #index e gore siralama
y_train = y_train.sort_index()

plt.plot(x_train, y_train)

plt.plot(x_test, y_pred)

















