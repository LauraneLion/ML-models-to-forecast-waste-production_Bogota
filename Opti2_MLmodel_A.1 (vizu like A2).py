# -*- coding: utf-8 -*-
"""
X = area, category of waste, date
f(xi)=y
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OrdinalEncoder, RobustScaler
from sklearn.linear_model import ElasticNet
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import StratifiedKFold, cross_val_score, ShuffleSplit
from sklearn.model_selection import learning_curve
from sklearn.metrics import mean_absolute_error, mean_squared_error

np.random.seed(0)  #fix the random state

"""
PreProcessing
"""
###Uploading of the Dataset
data = pd.read_excel('Dataset18_24.xlsx') 
#print(data.head())

###Creation of variables
X = data[['date','area','category of waste']].values
y = data['waste generation'].values 

y = y.reshape((y.shape[0],1))   #default value : (1129,)
#print(y.shape, X.shape)


###Encoding
encoder = OrdinalEncoder()
X = encoder.fit_transform(X)
print(X)


###Scaling
scaler = RobustScaler()
y = scaler.fit_transform(y)
#print(y)


###Creation of Trainset and Testset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
#print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
#print(X_test)



"""
#Modelling
"""
#RandomForest
forest = RandomForestRegressor()
forest.fit(X_train, y_train.ravel())
print("R2_forest=",forest.score(X_train, y_train.ravel()))

predictions_forest = forest.predict(X_test)

plt.figure()
plt.scatter(X_train[:,0], y_train)
plt.scatter(X_test[:,0], y_test)
plt.scatter(X_test[:,0], predictions_forest, c='r')
plt.title('random forest non opt')
plt.xlabel('date')
plt.ylabel('waste generation (in tons)')

plt.figure()
plt.scatter(X_train[:,1], y_train)
plt.scatter(X_test[:,1], y_test)
plt.scatter(X_test[:,1], predictions_forest, c='r')
plt.title('random forest non opt')
plt.xlabel('ASE')
plt.ylabel('waste generation (in tons)')

plt.figure()
plt.scatter(X_train[:,2], y_train)
plt.scatter(X_test[:,2], y_test)
plt.scatter(X_test[:,2], predictions_forest, c='r')
plt.title('random forest non opt')
plt.xlabel('category')
plt.ylabel('waste generation (in tons)')



"""Optimization"""

###CrossValidation
cv = ShuffleSplit(3, test_size=0.2) 


###Optimisation of hyperparameters
#https://www.stochasticbard.com/blog/lasso_regression/
def construct_model(X_train, y_train):
    regression_models = [RandomForestRegressor()]
    forest_parameters = {'n_estimators': np.arange(10, 100, 10), 'criterion':['squared_error', 'friedman_mse', 'absolute_error']} #'random_state':[123]}
    parameters = [forest_parameters]
    estimators = []
           
        # iterate through each classifier and use GridSearchCV
    for i,model in enumerate(regression_models):
        grid = GridSearchCV(model,
                            param_grid = parameters[i], # hyperparameters
                            cv=3)
        grid.fit(X_train, y_train)
        estimators.append([grid.best_params_, grid.best_score_, grid.best_estimator_])
    
    return estimators

estim = construct_model(X_train, y_train.ravel())
print(estim)


###Saving the new optimized model
forest_op = estim[0][2]

forest_op.fit(X_train, y_train.ravel())
predictions_forest_op = forest_op.predict(X_test)

#print("R2_tree_opt=",tree_op.score(X_train, y_train))
#print("MAE_tree_opt=", mean_absolute_error(y_test, predictions_tree_op), " and MSE_tree_opt=", mean_squared_error(y_test, predictions_tree_op))


###graf optimized model
plt.figure()
plt.scatter(X_train[:,0], y_train)
plt.scatter(X_test[:,0], y_test)
plt.scatter(X_test[:,0], predictions_forest_op, c='r')
plt.xlabel('date')
plt.ylabel('waste generation (in tons)')
plt.title('Random forest opt')


###vizualisation of each area and category
X_train_df = pd.DataFrame(X_train)    #conversion in dataframe because fit_transform return an array
y_train_df = pd.DataFrame(y_train)
X_test_df = pd.DataFrame(X_test)
y_test_df = pd.DataFrame(y_test)
y_pred_df = pd.DataFrame(predictions_forest_op)

df_train = pd.concat([y_train_df, X_train_df], axis=1)   #the same as data, but with encoding and scaling
df_train.columns = ['waste', 'date', 'area', 'category']
df_test = pd.concat([y_test_df, X_test_df], axis=1)
df_test.columns = ['waste', 'date', 'area', 'category']
df_pred = pd.concat([y_pred_df, X_test_df], axis=1)
df_pred.columns = ['waste', 'date', 'area', 'category']


ASE00_train = df_train[(df_train['area']==0) & (df_train['category']==0)]
ASE00_test = df_test[(df_test['area']==0) & (df_test['category']==0)]
ASE00_pred = df_pred[(df_pred['area']==0) & (df_pred['category']==0)]

ASE10_train = df_train[(df_train['area']==1) & (df_train['category']==0)]
ASE10_test = df_test[(df_test['area']==1) & (df_test['category']==0)]
ASE10_pred = df_pred[(df_pred['area']==1) & (df_pred['category']==0)]

ASE20_train = df_train[(df_train['area']==2) & (df_train['category']==0)]
ASE20_test = df_test[(df_test['area']==2) & (df_test['category']==0)]
ASE20_pred = df_pred[(df_pred['area']==2) & (df_pred['category']==0)]

ASE30_train = df_train[(df_train['area']==3) & (df_train['category']==0)]
ASE30_test = df_test[(df_test['area']==3) & (df_test['category']==0)]
ASE30_pred = df_pred[(df_pred['area']==3) & (df_pred['category']==0)]

ASE40_train = df_train[(df_train['area']==4) & (df_train['category']==0)]
ASE40_test = df_test[(df_test['area']==4) & (df_test['category']==0)]
ASE40_pred = df_pred[(df_pred['area']==4) & (df_pred['category']==0)]


plt.figure()
plt.scatter(ASE00_train['date'], ASE00_train['waste'], label='train')
plt.scatter(ASE00_test['date'], ASE00_test['waste'], label='test')
plt.scatter(ASE00_pred['date'], ASE00_pred['waste'], c='r', label='pred')
plt.legend()
plt.xlabel('date')
plt.ylabel('waste generation (in tons)')
plt.title('ASE1_Cat0 Random forest_opt')
plt.show()

plt.scatter(ASE10_train['date'], ASE10_train['waste'], label='train')
plt.scatter(ASE10_test['date'], ASE10_test['waste'], label='test')
plt.scatter(ASE10_pred['date'], ASE10_pred['waste'], c='r', label='pred')
plt.xlabel('date')
plt.ylabel('waste generation (in tons)')
plt.title('ASE2_Cat0 Random forest_opt')
plt.legend()
plt.show()



###Learning curve
"""
N_tree, train_score_tree, val_score_tree = learning_curve(tree, X_train00, y_train00.ravel(), train_sizes=np.linspace(0.5, 1.0, 5), cv=2)

plt.figure()
plt.plot(N_tree, train_score_tree.mean(axis=1), label='tree')
plt.legend()
plt.title('Training curves')

plt.figure()
plt.plot(N_tree,val_score_tree.mean(axis=1), label='tree')
plt.legend()
plt.title('Validation curves')
"""



