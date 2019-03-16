# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 22:13:32 2019

@author: yagmur
"""

import pandas as pd #veri çekmek,gerekli veriyi okumak için kullanılır.
import numpy as np #matematik kütüphanesidir.
import matplotlib.pyplot as plt   #verileri çizdirmek için kullanılır.
import seaborn as sns #Seaborn, Python'da istatistiksel grafikler yapmak için bir kütüphanedir. Matplotlib'in üzerine inşa edilmiştir ve pandaların veri yapılarıyla yakından bütünleşmiştir.Farklı tür bağımlı değişkenler için otomatik regresyon ve lineer regresyon modellerinin çizilmesini sağlaması özelliklerinden biridir
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression #model linear regression ile yapılacaktır
lm = LinearRegression()

USAhousing = pd.read_csv('USA_Housing.csv')# veri setini okudu.
USAhousing.head()  
USAhousing.info()
USAhousing.describe() # sayıları, ortalamaları, minimum ve maksimum değerleri ve bazı yüzdelikleri içerir.
USAhousing.columns #bulunan sütunların adlarını gösterir.
sns.pairplot(USAhousing)# seaborn ile çizdirme işlemini yapar.

sns.distplot(USAhousing['Price'])#price değişkenine göre grafiği çizdirir.
USAhousing.corr()
X = USAhousing[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms', #verileri, üzerinde çalışacak özellikleri içeren bir X dizisine ve y dizisine atar.
               'Avg. Area Number of Bedrooms', 'Area Population']]
y = USAhousing['Price'] 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101) #Amacımız, yeni verilere iyi yayılan bir model oluşturmaktır. Test setimiz yeni veriler için bir vekil olarak hizmet eder. Eğitilen veriler, doğrusal regresyon algoritmasını uyguladığımız verilerdir. Sonunda bu algoritmayı test verileri üzerinde test ediyoruz. Bölme kodu aşağıdaki gibidir: verilerin% 40'ının test verilerine gittiğini ve gerisinin eğitim setinde kaldığını söyleyebiliriz.
lm.fit(X_train,y_train)# x eksenine X_train i ve y eksenine y_train i oturtma işini yapar
predictions = lm.predict(X_test) #X_test e göre çıkacak olan sonuçtur
plt.scatter(y_test,predictions) #y_test verilerine göre predictions grafiğinin çizilmesi





