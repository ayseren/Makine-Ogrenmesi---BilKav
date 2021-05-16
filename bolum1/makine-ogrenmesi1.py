# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

#ikisi de olur
#pd.read_csv("C:/Users/Ayse/Desktop/calismalar/python ile makine ogrenmesi/veriler.csv")
veriler = pd.read_csv('veriler.csv')

#print(veriler[['boy']])
#print(veriler['boy']) seklinde yazinca baslikta boy yazmiyor

#sag taraftaki variable explorerda, tanimlanan degiskenler cikiyor


#################################### EKSIK VERILER ######################################################

eksikVeriler = pd.read_csv('eksik.csv')

#yas kolonunda eksik veri var
#tum yasin ortalamasini alip eksik olan yerlere bu degeri yazacak

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
Yas = eksikVeriler.iloc[:,1:4].values
#locIleYas = eksikVeriler.loc[:,'yas']
#print(locIleYas)
#print(Yas)

#loc ve iloc var
#loc ile etiket kullananarak veriye ulasilir (satir veya sutun isimlerine gore)
#iloc ile satır ve sütün index numarası ile verilerimize ulasilir

imputer = imputer.fit(Yas[:,1:4])
#fit fonk. ogrenilecek olan deger
#ogrenecegi sey de uc kolonun ortalama degerleri

Yas[:,1:4] = imputer.transform(Yas[:,1:4])
#nan degerlere transform fonk ile ortalama deger yazdirmak
#print(Yas)


#################################### KATEGORIK VERILER ######################################################

"""
            Veri Tipleri Arasinda Donusum
            
Kategorik ve sayisal olarak veriler ikiye ayrilir. Bir kisinin 
kadin veya erkek olmasi kategoriktir. Mesela kadin erkekten 
buyuk degil. Sayisal verilerde yas, kilo, maas gibi veriler.

Kategorik veriler de Nominal ve Ordinal olarak ikiye bolunur.
Ordinal veriler buyuktur kucuktur iliskisine girip olculemez.
Kapi numarasi siralanabilir ama olculenemez.
Nominal olanlarda siralama imkani da olculme imkani da yok 
mesela telefon markasi

Sayisal veriler de Ratio ve Interval olarak ikiye ayrilir.
Ratio orantalanabilen carpma bolme yapilabilen. Mesela yas
Interval toplama cikarma yapilabilen. Mesela hava sicakligi

    Kategorik Verileri Sayiya Cevirme
    
Kendi yorumum: bool halde yani 1 0 seklinde sayisal veri kullanmayi
ikili durumlarda kullanmak lazim ancak coklu durumlarda her etiketi
kolon basligina koyup 1 0 yapmak. deger kolon basligi tasiyorsa 1; 
tasimiyorsa 0 koyuyoruz. Sanirim kismen dogru oldu :D 

"""

Ulke = eksikVeriler.iloc[:,0:1].values
#print(Ulke)

labelencoder = preprocessing.LabelEncoder()
Ulke[:,0] = labelencoder.fit_transform(eksikVeriler.iloc[:,0])
#oncekinde fit ve transform ayri ayri cagirmistik 
#burda beraber cagiriyoruz
#print(Ulke)

onehotencoder = preprocessing.OneHotEncoder() 
Ulke = onehotencoder.fit_transform(Ulke).toarray()
#print(Ulke)


#################################### VERILERIN BIRLESTRILMESI ######################################################
#################################### DATAFRAME OLUSTURULMASI ######################################################

"""
    Eksik verilerin ortalama degere esitlenmesi ve
    ulkelerin encode edilmesi verileri birlestirecegiz
"""

#dataFramelerin index kolonu vardir. DataFramelerin diziden en buyuk farklari
#kolon isimleri ve index degerlerinin olmasidir

sonuc = pd.DataFrame(data=Ulke, index=range(22), columns=['fr', 'tr', 'us'])
#print(sonuc)

sonuc2 = pd.DataFrame(data=Yas, index=range(22), columns=['boy','kilo','yas'])

Cinsiyet = eksikVeriler.iloc[:,-1].values
#print(Cinsiyet)
#ilocta -1 koymak sondan bir kolonu al anlaminda

#values ile aldigimizdan oncekiler gibi bu da dizi,
#Cinsiyeti de DataFrame e donusturecegiz

sonuc3 = pd.DataFrame(data=Cinsiyet, index=range(22), columns=['cinsiyet'])

#### sonuclari birlestirme ####

#s = pd.concat([sonuc,sonuc2]) 
#bu sekilde birlesme dikey boyutta, axis = 0 olmus oluyor

s = pd.concat([sonuc, sonuc2], axis=1) 
#axis=1 denilince alt alta degil yan yana birlestirme yapmis oluyoruz

s2 = pd.concat([s, sonuc3], axis=1)
#print(s2)


#################################### VERININ TEST VE TRAIN OLARAK BOLUNMESI ######################################################

x_train, x_test, y_train, y_test = train_test_split(s, sonuc3, test_size=0.33, random_state=0)
#x bagimsiz degisken, y bagimli degisken
#x=s, y=sonuc3, random_state ogrennnnn


#################################### OZNITELIK OLCEKLEME ######################################################

#sayilar birbirine cok uzak sayilari yakinlastiriyoruz

standardScaler = StandardScaler()

X_Train = standardScaler.fit_transform(x_train)
Y_Test = standardScaler.fit_transform(x_test)
















