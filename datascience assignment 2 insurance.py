# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 10:39:29 2023

@author: admin
"""
##########Author- Hardik yadav############
##########BCA(datascience),A-20###########

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data=pd.read_csv('insurance.csv')
data.columns
data.info()
data.describe()

##########################################

sns.heatmap(data.corr(),annot=True)

#age vs charges

sns.scatterplot(x=data['age'],y=data['charges'])
sns.scatterplot(x=data['bmi'],y=data['charges'])

#gender vs chrges
sns.boxplot(x=data['sex'],y=data['charges'])

#children vs charges
sns.boxplot(x=data['children'],y=data['charges'])

#smoker versus cahrges
sns.boxplot(x=data['smoker'],y=data['charges'])

#region vs charges
sns.boxplot(x=data['region'],y=data['charges'])

###################################################

#convert categorical to numeral
columns=['sex','smoker','region']
from sklearn.preprocessing import LabelEncoder
encoder=LabelEncoder()
for column in columns:
    data[column]=encoder.fit_transform(data[column])


####################################################

x=data.drop(['charges'],axis=1)
y=data['charges']
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,
                        test_size=0.25,random_state=0)

#####################################################

#create a model
from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectFromModel
sel=SelectFromModel(Lasso(alpha=0.05))
sel.fit(x_train,y_train)
sel.get_support()
x.columns[sel.get_support()]


