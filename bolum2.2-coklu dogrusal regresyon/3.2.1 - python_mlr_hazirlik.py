# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 18:50:13 2020

@author: sadievrenseker
"""

"""
        COKLU DEGISKEN SECIMI
    Bagimli degiskeni tum bagimsiz degiskenler ayni oranda mi etkliler?
    
    1.Butun Degiskenleri Dahil Etme
    2.Geriye Dogru Eleme (Backward Elimination)
    3.Ileri Secim (Forward Selection)
    4.Iki Yonlu Eleme (Bidirectional Elimination)
    5.Skor Karsilastirmasi (Score Comparison)
    
    1 ile 5 deneme yanilma ile de bulunabilir; ancak 2,3,4
adim adim karsilastirma yontemleridir (stepwise).

    1.
    - Degisken secimi yapilmissa ve degiskenlerden eminsek
    - Zorunluluk varsa
    - Diger 4 yontemi kullanmadan once on fikir olmasi icin
kullanilir.

    2.
    - Cok kullanilir
    - Significant Level genelde 0.05 secilir.
    - Baslangicta tum degiskenler dahil edilir
    - p-value degeri en yuksek olan degisken ele alinir.
    - p > sl ise p degeri en yuksek olan degisken sistemden kaldirilir
    - makine ogrenmesi guncellenir ve diger en yuksek p degeri olan
degisken ele alinarak islemler tekrar edilir.
    - bu islem p < sl olana kadar devam eder.
    
    3.
    - 2.de en yuksek p value aliniyordu. Burda da en dusuk p value
alinarak 2.deki benzer islemler yapiliyor. 
    - Burada degisken kaldirilmaz yeni degisken eklenir.
(2.nin komple ziddi)

    4.
    - 2. ile 3. nun birlesimi
    - Sl iki tane olabilir. (ileriye icin ve geriye icin birer tane)
    - Butun degiskenlerle model insa edilir.
    - En dusuk p-value degerine sahip degisken ele alinir.
    - Diger butun degiskenler de sisteme dahil edilir.
    - Sl degerinin altinda olan degiskenler sistemde kalir ve
eski degiskenlerden hicbirisi sistemden cikarilmaz.
    - Yani iki kriterimiz var biri yeni bir degiskeni sisteme ekleme
kriteri, digeri bir degiskenin sistemde kalma kriteri

    5.
    - Basari kriteri belirlenir sl gibi.
    - Butun olasi regresyon modelleri insa edilir (ikili secim olur).
    - Basta belirlenen kriteri en iyi saglayan yontem secilir.
    
    Bu coklu degisken secimleri sadece regresyonda kullanilmaz. Geneldir.
"""

#1.kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

veriler = pd.read_csv('veriler.csv')

#YAS
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

yas = veriler.iloc[:,1:4].values
imputer = imputer.fit(yas[:, 1:4])
yas[:,1:4]=imputer.transform(yas[:,1:4])

#ULKE encoder: Kategorik -> Numeric
ulke = veriler.iloc[:,0:1].values

from sklearn import preprocessing

le = preprocessing.LabelEncoder()

ulke[:,0] = le.fit_transform(veriler.iloc[:,0])
#print(ulke)

ohe = preprocessing.OneHotEncoder()
ulke = ohe.fit_transform(ulke).toarray()

#CINSIYET encoder: Kategorik -> Numeric
cinsiyet = veriler.iloc[:,-1:].values

from sklearn import preprocessing

le = preprocessing.LabelEncoder()

cinsiyet[:,-1] = le.fit_transform(veriler.iloc[:,-1])

ohe = preprocessing.OneHotEncoder()
cinsiyet = ohe.fit_transform(cinsiyet).toarray()

print(cinsiyet)


#numpy dizileri dataframe donusumu
sonuc = pd.DataFrame(data=ulke, index = range(22), columns = ['fr','tr','us'])
print(sonuc)

sonuc2 = pd.DataFrame(data=yas, index = range(22), columns = ['boy','kilo','yas'])
print(sonuc2)

sonuc3 = pd.DataFrame(data = cinsiyet[:,:1], index = range(22), columns = ['cinsiyet'])
#cinsiyet[:,:1] diyere dummy variable tuzagindan kurtuluyoruz. (sadece 1 kolonu al)
print(sonuc3)


#dataframe birlestirme islemi
s=pd.concat([sonuc,sonuc2], axis=1)
print(s)

s2=pd.concat([s,sonuc3], axis=1)
print(s2)


#verilerin egitim ve test icin bolunmesi
from sklearn.model_selection import train_test_split

x_train, x_test,y_train,y_test = train_test_split(s,sonuc3,test_size=0.33, random_state=0)


############################# Coklu Degisken Lineer Model Olusturma #############################

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(x_train, y_train) #x_trainden y_traini ogren

y_pred = regressor.predict(x_test)
#x_testi yukarda ogrendigim makine ogrenme algoritmasina gore predict et ve y_prede yazdir


############################## boy kolonunu tahmin etme

boy = s2.iloc[:3:4].values

sol = s2.iloc[:,:3]
sag = s2.iloc[:,4:]
boysuzVeri = pd.concat([sol,sag],axis=1)

"""
x_train, x_test,y_train,y_test = train_test_split(boysuzVeri,boy,test_size=0.33, random_state=0)


regressor2 = LinearRegression()
regressor2.fit(x_train, y_train) #x_trainden y_traini ogren

y_pred = regressor2.predict(x_test)

"""


############################# Backward Elimination #############################

import statsmodels.formula.api as smf
import statsmodels.api as sm

X = np.append(arr = np.ones((22,1)).astype(int), values = boysuzVeri, axis = 1)
#dogru denklemindeki sabit deger su anda bizim veri setimizde yok
#veri setine 1e esit olan sabit deger her satira eklendi

"""
#ilkini yorumda buldugum sekilde cozdum ancak ise yaramadi
#hata aliyorum

X_liste = boysuzVeri.iloc[:,[0,1,2,3,4,5]].values
X_liste = np.array(X_liste, dtype=float)
model = sm.OLS(endog = boy,exog = X_liste)
#endog digerleri ile baglantisi kuracagimiz;
#exog baglantisini kurmak istedigimiz deger
#yani endog degerine alacagimiz degeri yazdik bulmak icin
#exog a yazdigimiz her sutunun endoga olan etkisini olmemizdir
r = model.fit()
print(r.summary())

X_liste = boysuzVeri.iloc[:,[0,1,2,3,5]].values #ne buyuk p value elendi
X_liste = np.array(X_liste, dtype=float)
model = sm.OLS(endog = boy,exog = X_liste).fit()
print(model.summary())

X_liste = boysuzVeri.iloc[:,[0,1,2,3]].values
X_liste = np.array(X_liste, dtype=float)
model = sm.OLS(endog = boy,exog = X_liste).fit()
print(model.summary())

"""

X_l = boysuzVeri.iloc[:,[0,1,2,3,4,5]].values
X_l = np.array(X_l,dtype=float)
model = sm.OLS(boy,X_l).fit()
print(model.summary())


X_l = boysuzVeri.iloc[:,[0,1,2,3,5]].values
X_l = np.array(X_l,dtype=float)
model = sm.OLS(boy,X_l).fit()
print(model.summary())



X_l = boysuzVeri.iloc[:,[0,1,2,3]].values
X_l = np.array(X_l,dtype=float)
model = sm.OLS(boy,X_l).fit()
print(model.summary())

