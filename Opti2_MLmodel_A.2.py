# -*- coding: utf-8 -*-
"""REG
y = waste generation
X = date
group = area & category of waste

f(X)=y   for each class
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OrdinalEncoder, RobustScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import ShuffleSplit, cross_val_score
from sklearn.model_selection import learning_curve
from sklearn.metrics import mean_absolute_error, mean_squared_error

np.random.seed(0)  #fix the random state


"""PreProcessing"""
###Uploading of the Dataset
data = pd.read_excel('Dataset18_24.xlsx') 
#print(data.head())

y = data['waste generation'].values
X = data[['date','area','category of waste']].values

y = y.reshape((y.shape[0],1))   #default value : (1129,)
print(y.shape, X.shape)


###Encoding
encoder = OrdinalEncoder()
X = encoder.fit_transform(X)
print(X)


###Scaling
scaler = RobustScaler()
y = scaler.fit_transform(y)
#print(y)


###Creation of the dataframes for each area and category
X_df = pd.DataFrame(X)    #conversion in dataframe because fit_transform return an array
y_df = pd.DataFrame(y)
df = pd.concat([y_df, X_df], axis=1)   #the same as data, but with encoding and scaling
df.columns = ['waste', 'date', 'area', 'category']
#print(df.head())

ASE0_0 = df[(df['area']==0) & (df['category']==0)]


###Creation of Trainset and Testset
X_train00, X_test00, y_train00, y_test00 = train_test_split(ASE0_0['date'], ASE0_0['waste'], test_size=0.2)
#print(X_train00.shape, X_test00.shape, y_train00.shape, y_test00.shape)

#Reshape because all the train and test set have shape = (n,)
X_train00 = X_train00.values.reshape((X_train00.shape[0],1)) 
y_train00 = y_train00.values.reshape((X_train00.shape[0],1))
X_test00 = X_test00.values.reshape((X_test00.shape[0],1)) 
y_test00 = y_test00.values.reshape((X_test00.shape[0],1))
#print(X_train00.shape, X_test00.shape, y_train00.shape, y_test00.shape)



"""Modelling"""
#KNN
knn = KNeighborsRegressor()
knn.fit(X_train00, y_train00)
#print("R2_knn=",knn.score(X_train00, y_train00))

predictions_knn = knn.predict(X_test00)
#print("MAE_knn=", mean_absolute_error(y_test00, predictions_knn), " and MSE_knn=", mean_squared_error(y_test00, predictions_knn))
plt.figure()
plt.scatter(X_train00, y_train00)
plt.scatter(X_test00, y_test00)
plt.scatter(X_test00, predictions_knn, c='r')
plt.title('KNN')


###SVR (kernel='rbf')
svr = SVR()
svr.fit(X_train00, y_train00.ravel())
#print("R2_svr=",svr.score(X_train00, y_train00))

predictions_svr = svr.predict(X_test00)
#print("MAE_svr=", mean_absolute_error(y_test00, predictions_svr), " and MSE_svr=", mean_squared_error(y_test00, predictions_svr))
plt.figure()
plt.scatter(X_train00, y_train00)
plt.scatter(X_test00, y_test00)
plt.scatter(X_test00, predictions_svr, c='r')
plt.title('SVR')
plt.xlabel('date')
plt.ylabel('waste generation (scaled)')


#DecisionTree
tree = DecisionTreeRegressor()
tree.fit(X_train00, y_train00.ravel())
#print('R2_tree=', tree.score(X_train00, y_train00))

predictions_tree = tree.predict(X_test00)
#print("MAE_tree=", mean_absolute_error(y_test00, predictions_tree), " and MSE_tree=", mean_squared_error(y_test00, predictions_tree))
plt.figure()
plt.scatter(X_train00, y_train00)
plt.scatter(X_test00, y_test00)
plt.scatter(X_test00, predictions_tree, c='r')
plt.title('tree')


#RandomForest
forest = RandomForestRegressor()
forest.fit(X_train00, y_train00.ravel())
#print("R2_forest=",forest.score(X_train00, y_train00))

predictions_forest = forest.predict(X_test00)
#print("MAE_forest=", mean_absolute_error(y_test00, predictions_forest), " and MSE_forest=", mean_squared_error(y_test00, predictions_forest))
plt.figure()
plt.scatter(X_train00, y_train00)
plt.scatter(X_test00, y_test00)
plt.scatter(X_test00, predictions_forest, c='r')
plt.title('forest')


#Polynomial
degree = PolynomialFeatures(degree=3, include_bias=False)
poly_features = degree.fit_transform(X_train00)
poly_test = degree.fit_transform(X_test00)

poly = LinearRegression()
poly.fit(poly_features, y_train00)
#print("R2_poly=",poly.score(poly_features, y_train00))

predictions_poly = poly.predict(poly_test)
#print("MAE_poly=", mean_absolute_error(y_test00, predictions_poly), " and MSE_poly=", mean_squared_error(y_test00, predictions_poly))
plt.figure()
plt.scatter(X_train00, y_train00)
plt.scatter(X_test00, y_test00)
plt.scatter(X_test00, predictions_poly, c='r')
plt.title('poly')



"""Optimization"""

###CrossValidation
cv = ShuffleSplit(3, test_size=0.2) 
# print('Mean CV_knn =', cross_val_score(knn, X_train00, y_train00.ravel(), cv=cv).mean())     #.ravel to avoid DataConversionWarning: 'A column-vector y was passed when a 1d array was expected'
# print('Mean CV_svr =', cross_val_score(svr, X_train00, y_train00.ravel(), cv=cv).mean())
# print('Mean CV_tree =', cross_val_score(tree, X_train00, y_train00.ravel(), cv=cv).mean())
# print('Mean CV_forest =', cross_val_score(forest, X_train00, y_train00.ravel(), cv=cv).mean())



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

estim = construct_model(X_train00, y_train00.ravel())
print(estim)


#optimization of polynomial hyperparameters
R2_poly = []
MAE_poly = []
MSE_poly = []
for i in range(1,16):
    degree = PolynomialFeatures(degree=i, include_bias=False)
    poly_features = degree.fit_transform(X_train00)
    poly_test = degree.fit_transform(X_test00)
    poly.fit(poly_features, y_train00)
    R2_poly.append(poly.score(poly_features, y_train00))
    predictions_poly = poly.predict(poly_test)
    MAE_poly.append(mean_absolute_error(y_test00, predictions_poly))
    MSE_poly.append(mean_squared_error(y_test00, predictions_poly))
#print(R2_poly.index(max(R2_poly)), MAE_poly.index(min(MAE_poly)), MSE_poly.index(min(MSE_poly)))

#graf to see the most efficient compromise
plt.figure()
plt.subplot(1,2,1)
plt.plot(np.arange(1,16), R2_poly)
plt.title('poly r2')
plt.subplot(1,2,2)
plt.plot(np.arange(1,16), MAE_poly, c='r')
plt.plot(np.arange(1,16), MSE_poly, c='g')
plt.title('poly error')

#ré-initialisation (for table of results)
degree = PolynomialFeatures(degree=3, include_bias=False)
poly_features = degree.fit_transform(X_train00)
poly_test = degree.fit_transform(X_test00)
poly.fit(poly_features, y_train00)
predictions_poly = poly.predict(poly_test)


###Saving the new optimized models
degree_op = PolynomialFeatures(degree=(R2_poly.index(max(R2_poly)))+1, include_bias=False)
poly_op = LinearRegression()
knn_op = estim[0][2]
svr_op = estim[1][2]
tree_op = estim[2][2]
forest_op = estim[3][2]

poly_features_op = degree_op.fit_transform(X_train00)
poly_test_op = degree_op.fit_transform(X_test00)
poly_op.fit(poly_features_op, y_train00)
predictions_poly_op = poly_op.predict(poly_test_op)
knn_op.fit(X_train00, y_train00)
predictions_knn_op = knn_op.predict(X_test00)
svr_op.fit(X_train00, y_train00.ravel())
predictions_svr_op = svr_op.predict(X_test00)
tree_op.fit(X_train00, y_train00.ravel())
predictions_tree_op = tree_op.predict(X_test00)
forest_op.fit(X_train00, y_train00.ravel())
predictions_forest_op = forest_op.predict(X_test00)


###Table of results
head = np.array([['state', 'metrics', 'KNN', 'SVR', 'DTR', 'RFR', 'Poly']])
#print(head.shape)

C1 = np.array([[i] for i in ['non-optimized', 'grid'] for j in range(3)])
#print(C1.shape)
C2 = np.array([[i] for j in range(2) for i in ['R2','MAE', 'MSE']])
#print(C2.shape)
C1_2 = np.concatenate((C1, C2), axis=1)
#print(C1_2.shape)

results = [[knn.score(X_train00, y_train00), svr.score(X_train00, y_train00), tree.score(X_train00, y_train00), forest.score(X_train00, y_train00)], 
           [mean_absolute_error(y_test00, i) for i in [predictions_knn, predictions_svr, predictions_tree, predictions_forest]],
           [mean_squared_error(y_test00, i) for i in [predictions_knn, predictions_svr, predictions_tree, predictions_forest]],
           [estim[i][1] for i in range(4)],
           [mean_absolute_error(y_test00, i) for i in [predictions_knn_op, predictions_svr_op, predictions_tree_op, predictions_forest_op]],
           [mean_squared_error(y_test00, i) for i in [predictions_knn_op, predictions_svr_op, predictions_tree_op, predictions_forest_op]]]
results = np.array(results)
results = np.round(results, 4)
#print(results.shape)

C_poly = [[np.round(poly.score(poly_features, y_train00), 4)],
          [np.round(mean_absolute_error(y_test00, predictions_poly), 4)], 
          [np.round(mean_squared_error(y_test00, predictions_poly), 4)], 
          [np.round(poly_op.score(poly_features_op, y_train00), 4)],
          [np.round(mean_absolute_error(y_test00, predictions_poly_op), 4)], 
          [np.round(mean_squared_error(y_test00, predictions_poly_op), 4)]]
C_poly = np.array(C_poly)

results = np.concatenate((results, C_poly), axis=1)
results = np.concatenate((C1_2, results), axis=1)
results = np.concatenate((head, results), axis=0)
results_df = pd.DataFrame(results)
print(results_df)

###saving the table of results in excel
excel = pd.ExcelWriter('resultsV2_18_24.xlsx')
results_df.to_excel(excel, sheet_name='Feuille1', index=False)
excel.save()


###graf optimized models
plt.figure(figsize = (15, 15))
plt.gcf().subplots_adjust(left = 0.3, bottom = 0.3,
                       right = 0.7, top = 0.7, wspace = 0.2, hspace = 0.7)
plt.subplot(3,2,1)
plt.scatter(X_train00, y_train00)
plt.scatter(X_test00, y_test00)
plt.scatter(X_test00, predictions_knn_op, c='r')
plt.xlabel('date')
plt.ylabel('waste generation (in tons)')
plt.title('KNN')
plt.subplot(3,2,2)
plt.scatter(X_train00, y_train00)
plt.scatter(X_test00, y_test00)
plt.scatter(X_test00, predictions_svr_op, c='r')
plt.xlabel('date')
plt.ylabel('waste generation (in tons)')
plt.title('SVR')
plt.subplot(3,2,3)
plt.scatter(X_train00, y_train00)
plt.scatter(X_test00, y_test00)
plt.scatter(X_test00, predictions_tree_op, c='r')
plt.xlabel('date')
plt.ylabel('waste generation (in tons)')
plt.title('Decision tree')
plt.subplot(3,2,4)
plt.scatter(X_train00, y_train00)
plt.scatter(X_test00, y_test00)
plt.scatter(X_test00, predictions_forest_op, c='r')
plt.xlabel('date')
plt.ylabel('waste generation (in tons)')
plt.title('Random forest')
plt.subplot(3,2,5)
plt.scatter(X_train00, y_train00)
plt.scatter(X_test00, y_test00)
plt.scatter(X_test00, predictions_poly_op, c='r')
plt.xlabel('date')
plt.ylabel('waste generation (in tons)')
plt.title('Polynomial')

#all predictions on the same graph -> not very readable
"""plt.figure()
plt.scatter(X_train00, y_train00)
plt.scatter(X_test00, y_test00)
plt.scatter(X_test00, predictions_knn_op, c='r', label='knn')
plt.scatter(X_test00, predictions_svr_op, c='g', label='svr')
plt.scatter(X_test00, predictions_tree_op, c='y', label='tree')
plt.scatter(X_test00, predictions_forest_op, c='m', label='forest')
plt.scatter(X_test00, predictions_poly_op, c='c', label='poly')
plt.legend()
plt.xlabel('date')
plt.ylabel('waste generation (in tons)')
plt.title('Polynomial')
"""


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





