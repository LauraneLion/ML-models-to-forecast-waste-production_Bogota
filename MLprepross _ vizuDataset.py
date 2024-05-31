# -*- coding: utf-8 -*-
"""
X = area, category of waste
f(xi)=y
"""
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, RobustScaler

"""
PreProcessing
"""
###Uploading of the Dataset
data = pd.read_excel('Dataset21_24.xlsx', index_col='date', parse_dates=True)   #transformation of the column 'date' in index 
#print(data.head())
print(data.describe())

###Visualization of the Dataset
data.reset_index().plot.scatter(x='date', y='waste generation')
data.reset_index().groupby('area').plot.scatter(x='date', y='waste generation')        #one graf per area
data.reset_index().groupby('category of waste').plot.scatter(x='date', y='waste generation')    #one graf per category


###Creation of variables
X = data[['area','category of waste']].values
y = data['waste generation'].values 

y = y.reshape((y.shape[0],1))   #default value : (1129,)
#print(y.shape, X.shape)


###Encoding
encoder = OrdinalEncoder()
X = encoder.fit_transform(X)
#print(X)

###Visualization of Dataset using Encoding
data.reset_index().plot.scatter(x='date', y='waste generation', c=X[:,0])
plt.title("color per area")
data.reset_index().plot.scatter(x='date', y='waste generation', c=X[:,1])
plt.title("color per category")

#graf 3D
plt.figure()
ax=plt.axes(projection='3d')       
ax.scatter(X[:,0], X[:,1], y, c=X[:,0])
plt.title("color per area")

plt.figure()
ax=plt.axes(projection='3d') 
ax.scatter(X[:,0], X[:,1], y, c=X[:,1])
plt.title("color per category")


###Scaling
scaler = RobustScaler()
y = scaler.fit_transform(y)
#print(y)


###Creation of Trainset and Testset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
#print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
#print(X_test)



