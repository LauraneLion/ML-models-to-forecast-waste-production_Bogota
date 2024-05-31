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
print(y.shape, X.shape)


###Encoding
encoder = OrdinalEncoder()
X = encoder.fit_transform(X)
print(X)
    #pb encodage date : devrait être de 0 à 62 !!!


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

###ElasticNet
elasticnet = ElasticNet(alpha=0.2, l1_ratio=0.5, random_state=123)
elasticnet.fit(X_train, y_train)
#print("R2_elasticnet=", elasticnet.score(X_train, y_train))

predictions_en = elasticnet.predict(X_test)

"""plt.scatter(X_train[:,0], y_train)
plt.scatter(X_test[:,0], y_test)
plt.scatter(X_test[:,0], predictions_en, c='r')"""

"""plt.scatter(X_train[:,1], y_train)
plt.scatter(X_test[:,1], y_test)
plt.scatter(X_test[:,1], predictions_en, c='r')"""


###SVR rbf
svr = SVR(C=100)
svr.fit(X_train, y_train.ravel())
#print("R2_SVR=", svr.score(X_train, y_train.ravel()))

predictions_svr = svr.predict(X_test)

"""plt.scatter(X_train[:,0], y_train)
plt.scatter(X_test[:,0], y_test)
plt.scatter(X_test[:,0], predictions_svr, c='r')"""

"""plt.scatter(X_train[:,1], y_train)
plt.scatter(X_test[:,1], y_test)
plt.scatter(X_test[:,1], predictions_svr, c='r')"""


#KNN
knn = KNeighborsRegressor(15)
knn.fit(X_train, y_train)
#print("R2_knn=",knn.score(X_train, y_train))

predictions_knn = knn.predict(X_test)
#print("MAE_knn=", mean_absolute_error(y_test, predictions_knn), " and MSE_knn=", mean_squared_error(y_test, predictions_knn))

"""plt.scatter(X_train[:,0], y_train)
plt.scatter(X_test[:,0], y_test)
plt.scatter(X_test[:,0], predictions_knn, c='r')"""

"""plt.scatter(X_train[:,1], y_train)
plt.scatter(X_test[:,1], y_test)
plt.scatter(X_test[:,1], predictions_knn, c='r')"""


#DecisionTree
tree = DecisionTreeRegressor(max_depth=10, min_samples_split=10)
tree.fit(X_train, y_train)
#print("R2_tree=",tree.score(X_train, y_train))

predictions_tree = tree.predict(X_test)
#print("MAE_tree=", mean_absolute_error(y_test, predictions_tree), " and MSE_tree=", mean_squared_error(y_test, predictions_tree))

"""plt.scatter(X_train[:,0], y_train)
plt.scatter(X_test[:,0], y_test)
plt.scatter(X_test[:,0], predictions_tree, c='r')"""

"""plt.scatter(X_train[:,1], y_train)
plt.scatter(X_test[:,1], y_test)
plt.scatter(X_test[:,1], predictions_tree, c='r')"""


#RandomForest
forest = RandomForestRegressor()
forest.fit(X_train, y_train.ravel())
#print("R2_forest=",forest.score(X_train, y_train.ravel()))

predictions_forest = forest.predict(X_test)

plt.figure()
plt.scatter(X_train[:,0], y_train)
plt.scatter(X_test[:,0], y_test)
plt.scatter(X_test[:,0], predictions_forest, c='r')
plt.title('random forest')
plt.xlabel('date')
plt.ylabel('waste generation (in tons)')

plt.figure()
plt.scatter(X_train[:,1], y_train)
plt.scatter(X_test[:,1], y_test)
plt.scatter(X_test[:,1], predictions_forest, c='r')
plt.title('random forest')
plt.xlabel('ASE')
plt.ylabel('waste generation (in tons)')

plt.figure()
plt.scatter(X_train[:,2], y_train)
plt.scatter(X_test[:,2], y_test)
plt.scatter(X_test[:,2], predictions_forest, c='r')
plt.title('random forest')
plt.xlabel('category')
plt.ylabel('waste generation (in tons)')


"""Optimization"""

###CrossValidation
cv = ShuffleSplit(3, test_size=0.2) 
# print('Mean CV_knn =', cross_val_score(knn, X_train, y_train.ravel(), cv=cv).mean())     #.ravel to avoid DataConversionWarning: 'A column-vector y was passed when a 1d array was expected'
# print('Mean CV_svr =', cross_val_score(svr, X_train, y_train.ravel(), cv=cv).mean())
# print('Mean CV_tree =', cross_val_score(tree, X_train, y_train.ravel(), cv=cv).mean())
# print('Mean CV_forest =', cross_val_score(forest, X_train, y_train.ravel(), cv=cv).mean())


###Optimisation of hyperparameters
#https://www.stochasticbard.com/blog/lasso_regression/
def construct_model(X_train, y_train):
    regression_models = [KNeighborsRegressor(), SVR(), DecisionTreeRegressor(), RandomForestRegressor()]  # the list of classifiers to use
    knn_parameters = {'n_neighbors': np.arange(1, 15, 1), 'weights':['uniform', 'distance'], 'p':[1, 2, 3]}
    svr_parameters = {'C': np.arange(50, 200, 10), 'kernel':['poly', 'rbf', 'sigmoid'], 'epsilon':np.arange(0.1, 0.5, 0.1)}
    tree_parameters = {'criterion':['squared_error', 'friedman_mse', 'absolute_error'], 'max_depth': np.arange(1, 10, 1)}  #'random_state':[123]}
    forest_parameters = {'n_estimators': np.arange(10, 100, 10), 'criterion':['squared_error', 'friedman_mse', 'absolute_error']} #'random_state':[123]}
        #criterion: not 'poisson' because we have negative values du to te scaling
    parameters = [knn_parameters, svr_parameters, tree_parameters, forest_parameters]
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


###Saving the new optimized models
knn_op = estim[0][2]
svr_op = estim[1][2]
tree_op = estim[2][2]
forest_op = estim[3][2]

knn_op.fit(X_train, y_train)
predictions_knn_op = knn_op.predict(X_test)
svr_op.fit(X_train, y_train.ravel())
predictions_svr_op = svr_op.predict(X_test)
tree_op.fit(X_train, y_train.ravel())
predictions_tree_op = tree_op.predict(X_test)
forest_op.fit(X_train, y_train.ravel())
predictions_forest_op = forest_op.predict(X_test)


###Table of results
head = np.array([['state', 'metrics', 'KNN', 'SVR', 'DTR', 'RFR']])
#print(head.shape)

C1 = np.array([[i] for i in ['non-optimized', 'grid'] for j in range(3)])
#print(C1.shape)
C2 = np.array([[i] for j in range(2) for i in ['R2','MAE', 'MSE']])
#print(C2.shape)
C1_2 = np.concatenate((C1, C2), axis=1)
#print(C1_2.shape)

results = [[knn.score(X_train, y_train), svr.score(X_train, y_train), tree.score(X_train, y_train), forest.score(X_train, y_train)], 
           [mean_absolute_error(y_test, i) for i in [predictions_knn, predictions_svr, predictions_tree, predictions_forest]],
           [mean_squared_error(y_test, i) for i in [predictions_knn, predictions_svr, predictions_tree, predictions_forest]],
           [estim[i][1] for i in range(4)],
           [mean_absolute_error(y_test, i) for i in [predictions_knn_op, predictions_svr_op, predictions_tree_op, predictions_forest_op]],
           [mean_squared_error(y_test, i) for i in [predictions_knn_op, predictions_svr_op, predictions_tree_op, predictions_forest_op]]]
results = np.array(results)
results = np.round(results, 4)
#print(results.shape)

results = np.concatenate((C1_2, results), axis=1)
results = np.concatenate((head, results), axis=0)
results_df = pd.DataFrame(results)
print(results_df)

###saving the table of results in excel
excel = pd.ExcelWriter('resultsV1.2.2_18-24.xlsx')
results_df.to_excel(excel, sheet_name='Feuille1', index=False)
excel.save()

###graf optimized models
plt.figure(figsize = (15, 10))
plt.gcf().subplots_adjust(left = 0.3, bottom = 0.3,
                       right = 0.7, top = 0.7, wspace = 0.2, hspace = 0.7)
plt.subplot(2,2,1)
plt.scatter(X_train[:,0], y_train)
plt.scatter(X_test[:,0], y_test)
plt.scatter(X_test[:,0], predictions_knn_op, c='r')
plt.xlabel('date')
plt.ylabel('waste generation (in tons)')
plt.title('KNN')
plt.subplot(2,2,2)
plt.scatter(X_train[:,0], y_train)
plt.scatter(X_test[:,0], y_test)
plt.scatter(X_test[:,0], predictions_svr_op, c='r')
plt.xlabel('date')
plt.ylabel('waste generation (in tons)')
plt.title('SVR')
plt.subplot(2,2,3)
plt.scatter(X_train[:,0], y_train)
plt.scatter(X_test[:,0], y_test)
plt.scatter(X_test[:,0], predictions_tree_op, c='r')
plt.xlabel('date')
plt.ylabel('waste generation (in tons)')
plt.title('Decision tree')
plt.subplot(2,2,4)
plt.scatter(X_train[:,0], y_train)
plt.scatter(X_test[:,0], y_test)
plt.scatter(X_test[:,0], predictions_forest_op, c='r')
plt.xlabel('date')
plt.ylabel('waste generation (in tons)')
plt.title('Random forest')


###Learning curves
"""
N_knn, train_score_knn, val_score_knn = learning_curve(knn, X_train00, y_train00.ravel(), train_sizes=np.linspace(0.5, 1.0, 5), cv=2)
N_svr, train_score_svr, val_score_svr = learning_curve(svr, X_train00, y_train00.ravel(), train_sizes=np.linspace(0.5, 1.0, 5), cv=2)
N_tree, train_score_tree, val_score_tree = learning_curve(tree, X_train00, y_train00.ravel(), train_sizes=np.linspace(0.5, 1.0, 5), cv=2)
N_forest, train_score_forest, val_score_forest = learning_curve(forest, X_train00, y_train00.ravel(), train_sizes=np.linspace(0.5, 1.0, 5), cv=2)

plt.figure()
plt.subplot(2,2,1)
plt.plot(N_knn,train_score_knn.mean(axis=1), label='knn')
plt.legend()
plt.title('Training curves')
plt.subplot(2,2,2)
plt.plot(N_svr,train_score_svr.mean(axis=1), label='svr')
plt.legend()
plt.subplot(2,2,3)
plt.plot(N_tree, train_score_tree.mean(axis=1), label='tree')
plt.legend()
plt.subplot(2,2,4)
plt.plot(N_forest, train_score_forest.mean(axis=1), label='forest')
plt.legend()

plt.figure()
plt.subplot(2,2,1)
plt.plot(N_knn,val_score_knn.mean(axis=1), label='knn')
plt.legend()
plt.title('Validation curves')
plt.subplot(2,2,2)
plt.plot(N_svr,val_score_svr.mean(axis=1), label='svr')
plt.legend()
plt.subplot(2,2,3)
plt.plot(N_tree,val_score_tree.mean(axis=1), label='tree')
plt.legend()
plt.subplot(2,2,4)
plt.plot(N_forest, val_score_forest.mean(axis=1), label='forest')
plt.legend()
"""



