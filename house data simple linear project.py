# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 22:35:52 2023

@author: MRUTYUNJAY
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
dataset=pd.read_csv(r'D:\class\13th march\SLR - Practicle\House_data.csv')
space=dataset['sqft_living']
price=dataset['price']
x=np.array(space).reshape(-1,1)
y=np.array(price)
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(xtrain,ytrain)
import warnings
warnings.filterwarnings('ignore')
pred=regressor.predict(xtest)
plt.scatter(xtrain,ytrain,color='red')
plt.plot(xtrain,regressor.predict(xtrain),color='blue')
plt.title("Visuals for Training Dataset")
plt.xlabel("Space")
plt.ylabel("Price")
plt.show()
plt.scatter(xtest,ytest,color='red')
plt.plot(xtrain,regressor.predict(xtrain),color='blue')
plt.title("Visuals for Test Dataset")
plt.xlabel("Space")
plt.ylabel("Price")
plt.show()
dataset.head()
dataset.info()
dataset.tail()
dataset.describe()
dataset.columns
