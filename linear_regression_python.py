import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression as lr
import matplotlib.pyplot as plt

data=pd.read_csv("linear.csv")
print(data) 
x=data["metrekare"] # veri setindeki metrekare x eksenine ait olacaktır.
y=data["fiyat"] #veri setindeki fiyat y eksenine ait olacaktır. 

x=x.values.reshape(99,1) # veri seti 99 satır 1 sütundan oluşmaktadır.
y=y.values.reshape(99,1)

lineerregresyon=lr()
lineerregresyon.fit(x,y) 
lineerregresyon.predict(x) #verilen metrekare değerine göre fiyatın yorumlanması
m=lineerregresyon.coef_ # katsayıyı verir.
b= lineerregresyon.intercept_ # ye de kesiştiği yer

a=np.arange(150)
plt.scatter(x,y)
plt.scatter(a,m*a+b)
plt.show()


for i in range(len(y)): #her y değeri(fiyat) için hata payı 
    hatakaresi=(y[i]-(m*x[i]+b))**2