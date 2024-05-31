# -*- coding: utf-8 -*-
"""
PREDICITON FUNCTION
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OrdinalEncoder, RobustScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

np.random.seed(0)  #fix the random state


"""
PreProcessing
"""
###Uploading of the Dataset
data = pd.read_excel('Dataset21_24_pred.xlsx') 
#print(data.head())

###Creation of variables
X = data[['date','area','category of waste']].values
y = data['waste generation'].values 

y = y.reshape((y.shape[0],1))   #default value : (1129,)
#print(y.shape, X.shape)


###Encoding
encoder = OrdinalEncoder()
X = encoder.fit_transform(X)
#print(X)


###Scaling
scaler = RobustScaler()
y = scaler.fit_transform(y)
#print(y)


###Creation of Trainset and Testset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
#print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
#print(X_test)

"""Modelling"""
forest = RandomForestRegressor()
forest.fit(X_train, y_train.ravel())

tree = DecisionTreeRegressor(max_depth=10, min_samples_split=10)
tree.fit(X_train, y_train)

###Optimisation of hyperparameters
#https://www.stochasticbard.com/blog/lasso_regression/
def construct_model(X_train, y_train):
    regression_models = [RandomForestRegressor()]  # the list of classifiers to use
    forest_parameters = {'n_estimators': np.arange(10, 100, 10), 'criterion':['squared_error', 'friedman_mse', 'absolute_error']} #'random_state':[123]}
        #criterion: not 'poisson' because we have negative values du to te scaling
    parameters = [forest_parameters]
    estimators = []
           
        # iterate through each classifier and use GridSearchCV
    for i,model in enumerate(regression_models):
        grid = GridSearchCV(model,
                            param_grid = parameters[i], # hyperparameters
                            cv=3)
        grid.fit(X_train, y_train)
        estimators.append([grid.best_params_, grid.best_score_, grid.best_estimator_])
    
    return estimators, grid.best_estimator_

estim, forest_op = construct_model(X_train, y_train.ravel())
print(estim)

forest_op.fit(X_train, y_train.ravel())


###vizualisation of each area and category
X_train_df = pd.DataFrame(X_train)    #conversion in dataframe because fit_transform return an array
y_train_df = pd.DataFrame(y_train)

df_train = pd.concat([y_train_df, X_train_df], axis=1)   #the same as data, but with encoding and scaling
df_train.columns = ['waste', 'date', 'area', 'category']

ASE00_train = df_train[(df_train['area']==0) & (df_train['category']==0)]
ASE10_train = df_train[(df_train['area']==1) & (df_train['category']==0)]
ASE20_train = df_train[(df_train['area']==2) & (df_train['category']==0)]
ASE30_train = df_train[(df_train['area']==3) & (df_train['category']==0)]
ASE40_train = df_train[(df_train['area']==4) & (df_train['category']==0)]


"""PRED FUNCTION"""
def pred(date, area, category):
    #X = data[['date','area','category of waste']]    #pb with my version of python to use input()
    # date = input('Which mounth? :')
    # area = input('Which area? :')
    # category = input('Which category? :')
    X_pred = np.array([[date, area, category]])
    y_pred = forest_op.predict(X_pred)
    y_pred=y_pred.reshape((y_pred.shape[0],1))
   
    if category==0:
        if area==0:
            plt.figure()
            plt.scatter(ASE00_train['date'], ASE00_train['waste'], label='train')
            plt.scatter(X_pred[:,0], y_pred, c='r', label='pred')
            plt.legend()
            plt.xlabel('date')
            plt.ylabel('waste generation (in tons)')
            plt.title('ASE1_Cat0 Random forest_opt')
            plt.show()
        
        elif area==1:
            plt.figure()
            plt.scatter(ASE10_train['date'], ASE10_train['waste'], label='train')
            plt.scatter(X_pred[:,0], y_pred, c='r', label='pred')
            plt.legend()
            plt.xlabel('date')
            plt.ylabel('waste generation (in tons)')
            plt.title('ASE1_Cat0 Random forest_opt')
            plt.show()
            
        elif area==2:
            plt.figure()
            plt.scatter(ASE20_train['date'], ASE20_train['waste'], label='train')
            plt.scatter(X_pred[:,0], y_pred, c='r', label='pred')
            plt.legend()
            plt.xlabel('date')
            plt.ylabel('waste generation (in tons)')
            plt.title('ASE1_Cat0 Random forest_opt')
            plt.show()
            
        elif area==3:
            plt.figure()
            plt.scatter(ASE30_train['date'], ASE30_train['waste'], label='train')
            plt.scatter(X_pred[:,0], y_pred, c='r', label='pred')
            plt.legend()
            plt.xlabel('date')
            plt.ylabel('waste generation (in tons)')
            plt.title('ASE1_Cat0 Random forest_opt')
            plt.show()
            
        elif area==4:
            plt.figure()
            plt.scatter(ASE40_train['date'], ASE40_train['waste'], label='train')
            plt.scatter(X_pred[:,0], y_pred, c='r', label='pred')
            plt.legend()
            plt.xlabel('date')
            plt.ylabel('waste generation (in tons)')
            plt.title('ASE1_Cat0 Random forest_opt')
            plt.show()
        
    return scaler.inverse_transform(y_pred)



def preds(date1, date2, area, category):
    X_pred = np.array([[date1, area, category]])
    for i in range(date1+1,date2+1):
        X_pred = np.vstack([X_pred, [i, area, category]])
               
    y_pred = forest_op.predict(X_pred)
    y_pred=y_pred.reshape((y_pred.shape[0],1))
    y_pred2 = tree.predict(X_pred)
    y_pred2=y_pred2.reshape((y_pred2.shape[0],1))
   
    if category==0:
        if area==0:
            plt.figure()
            plt.scatter(ASE00_train['date'], ASE00_train['waste'], label='train')
            plt.scatter(X_pred[:,0], y_pred, c='r', label='pred')
            plt.legend()
            plt.xlabel('date')
            plt.ylabel('waste generation (in tons)')
            plt.title('ASE1_Cat0 Random forest_opt')
            plt.show()
        
        elif area==1:
            plt.figure()
            plt.scatter(ASE10_train['date'], ASE10_train['waste'], label='train')
            plt.scatter(X_pred[:,0], y_pred, c='r', label='pred')
            plt.legend()
            plt.xlabel('date')
            plt.ylabel('waste generation (in tons)')
            plt.title('ASE1_Cat0 Random forest_opt')
            plt.show()
            
        elif area==2:
            plt.figure()
            plt.scatter(ASE20_train['date'], ASE20_train['waste'], label='train')
            plt.scatter(X_pred[:,0], y_pred, c='r', label='pred')
            plt.legend()
            plt.xlabel('date')
            plt.ylabel('waste generation (in tons)')
            plt.title('ASE1_Cat0 Random forest_opt')
            plt.show()
            
        elif area==3:
            plt.figure()
            plt.scatter(ASE30_train['date'], ASE30_train['waste'], label='train')
            plt.scatter(X_pred[:,0], y_pred, c='r', label='pred')
            plt.legend()
            plt.xlabel('date')
            plt.ylabel('waste generation (in tons)')
            plt.title('ASE1_Cat0 Random forest_opt')
            plt.show()
            
        elif area==4:
            plt.figure()
            plt.scatter(ASE40_train['date'], ASE40_train['waste'], label='train')
            plt.scatter(X_pred[:,0], y_pred, c='r', label='pred')
            plt.legend()
            plt.xlabel('date')
            plt.ylabel('waste generation (in tons)')
            plt.title('ASE1_Cat0 Random forest_opt')
            plt.show()
        
    
    return scaler.inverse_transform(y_pred), scaler.inverse_transform(y_pred2)



