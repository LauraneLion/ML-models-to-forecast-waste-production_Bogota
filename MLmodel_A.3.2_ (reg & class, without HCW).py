# -*- coding: utf-8 -*-
""" REG + CLAS
yreg = waste generation
yclass = category of waste
Xreg = date
Xclass = date, waste
group = area 

fi(Xi)=yi   for each class,   with i e [reg, class]
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OrdinalEncoder, RobustScaler
from sklearn import linear_model
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import ShuffleSplit, cross_val_score
from sklearn.model_selection import learning_curve
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score

np.random.seed(0)  #fix the random state


"""PreProcessing"""

###Uploading of the Dataset
data = pd.read_excel('Dataset21_24 - wthout_HCW.xlsx') 
#print(data.head())

yreg = data['waste generation'].values
yclas = data['category of waste'].values
X = data[['date','area']].values

yreg = yreg.reshape((yreg.shape[0],1))   #default value : (1129,)
yclas = yclas.reshape((yclas.shape[0],1))
#print(yreg.shape, yclas.shape, X.shape)


###Encoding
encoder = OrdinalEncoder()
X = encoder.fit_transform(X)
yclas = encoder.fit_transform(yclas)
#print(X, yclas)


###Scaling
scaler = RobustScaler()
yreg = scaler.fit_transform(yreg)
#print(yreg)


###Creation of the dataframes for each area and category
X_df = pd.DataFrame(X)    #conversion in dataframe because fit_transform return an array
yreg_df = pd.DataFrame(yreg)
yclas_df = pd.DataFrame(yclas)

df = pd.concat([yreg_df, yclas_df, X_df], axis=1)   #the same as data, but with encoding and scaling
df.columns = ['waste', 'category', 'date', 'area']
#print(df.head())

ASE0 = df[(df['area']==0)]  #split datatset
ASE1 = df[(df['area']==1)]
ASE2 = df[(df['area']==2)]
ASE3 = df[(df['area']==3)]
ASE4 = df[(df['area']==4)]


###Creation of Trainset and Testset
    #for regression
X_train0, X_test0, y_train0, y_test0 = train_test_split(ASE0['date'], ASE0['waste'], test_size=0.2)
X_train1, X_test1, y_train1, y_test1 = train_test_split(ASE1['date'], ASE1['waste'], test_size=0.2)
X_train2, X_test2, y_train2, y_test2 = train_test_split(ASE2['date'], ASE2['waste'], test_size=0.2)
X_train3, X_test3, y_train3, y_test3 = train_test_split(ASE3['date'], ASE3['waste'], test_size=0.2)
X_train4, X_test4, y_train4, y_test4 = train_test_split(ASE4['date'], ASE4['waste'], test_size=0.2)
#print(X_train0.shape, X_test0.shape, y_train0.shape, y_test0.shape)

    #for classification
X_trainA, X_testA, y_trainA, y_testA = train_test_split(ASE0[['date', 'waste']], ASE0['category'], test_size=0.2)
X_trainB, X_testB, y_trainB, y_testB = train_test_split(ASE1[['date', 'waste']], ASE1['category'], test_size=0.2)
X_trainC, X_testC, y_trainC, y_testC = train_test_split(ASE2[['date', 'waste']], ASE2['category'], test_size=0.2)
X_trainD, X_testD, y_trainD, y_testD = train_test_split(ASE3[['date', 'waste']], ASE3['category'], test_size=0.2)
X_trainE, X_testE, y_trainE, y_testE = train_test_split(ASE4[['date', 'waste']], ASE4['category'], test_size=0.2)
#print(X_trainA.shape, X_testA.shape, y_trainA.shape, y_testA.shape)


#Reshape those with shape = (n,)
X_train0 = X_train0.values.reshape((X_train0.shape[0],1)) 
X_test0 = X_test0.values.reshape((X_test0.shape[0],1)) 
y_train0 = y_train0.values.reshape((y_train0.shape[0],1)) 
y_test0 = y_test0.values.reshape((y_test0.shape[0],1))
X_train1 = X_train1.values.reshape((X_train1.shape[0],1)) 
X_test1 = X_test1.values.reshape((X_test1.shape[0],1)) 
y_train1 = y_train1.values.reshape((y_train1.shape[0],1)) 
y_test1 = y_test1.values.reshape((y_test1.shape[0],1))
X_train2 = X_train2.values.reshape((X_train2.shape[0],1)) 
X_test2 = X_test2.values.reshape((X_test2.shape[0],1)) 
y_train2 = y_train2.values.reshape((y_train2.shape[0],1)) 
y_test2 = y_test2.values.reshape((y_test2.shape[0],1)) 
X_train3 = X_train3.values.reshape((X_train3.shape[0],1)) 
X_test3 = X_test3.values.reshape((X_test3.shape[0],1)) 
y_train3 = y_train3.values.reshape((y_train3.shape[0],1)) 
y_test3 = y_test3.values.reshape((y_test3.shape[0],1))
X_train4 = X_train4.values.reshape((X_train4.shape[0],1)) 
X_test4 = X_test4.values.reshape((X_test4.shape[0],1)) 
y_train4 = y_train4.values.reshape((y_train4.shape[0],1)) 
y_test4 = y_test4.values.reshape((y_test4.shape[0],1))
#print(X_train0.shape, X_test0.shape, y_train0.shape, y_test0.shape)

y_trainA = y_trainA.values.reshape((y_trainA.shape[0],1)) 
y_testA = y_testA.values.reshape((y_testA.shape[0],1))
y_trainB = y_trainB.values.reshape((y_trainB.shape[0],1)) 
y_testB = y_testB.values.reshape((y_testB.shape[0],1))
y_trainC = y_trainC.values.reshape((y_trainC.shape[0],1)) 
y_testC = y_testC.values.reshape((y_testC.shape[0],1)) 
y_trainD = y_trainD.values.reshape((y_trainD.shape[0],1)) 
y_testD = y_testD.values.reshape((y_testD.shape[0],1))
y_trainE = y_trainE.values.reshape((y_trainE.shape[0],1)) 
y_testE = y_testE.values.reshape((y_testE.shape[0],1))
#print(X_trainA.shape, X_testA.shape, y_trainA.shape, y_testA.shape)



"""Modelling"""

####REGRESSION
###Lasso
"""lasso = linear_model.Lasso(alpha=0.6)
lasso.fit(X_train0, y_train0)
print("R2_lasso=",lasso.score(X_train0, y_train0))
predictions_lasso = lasso.predict(X_test0)
plt.figure()
plt.scatter(X_train0['date'], y_train0)
plt.scatter(X_test0['date'], y_test0)
plt.scatter(X_test0['date'], predictions_lasso, c='r')
plt.title('Lasso')


###ElasticNet
elasticnet = linear_model.ElasticNet(alpha=0.2, l1_ratio=0.5)
elasticnet.fit(X_train0, y_train0)
print("R2_elasticnet=", elasticnet.score(X_train0, y_train0))
predictions_en = elasticnet.predict(X_test0)
plt.figure()
plt.scatter(X_train0['date'], y_train0)
plt.scatter(X_test0['date'], y_test0)
plt.scatter(X_test0['date'], predictions_en, c='r')
plt.title('ElasticNet')"""


#KNN
knn = KNeighborsRegressor()
knn.fit(X_train0, y_train0)
#print("R2_knn=",knn.score(X_train0, y_train0))

predictions_knn = knn.predict(X_test0)
plt.figure()
plt.scatter(X_train0, y_train0)
plt.scatter(X_test0, y_test0)
plt.scatter(X_test0, predictions_knn, c='r')
plt.title('KNN')


###SVR (kernel='rbf')
svr = SVR(C=100)
svr.fit(X_train0, y_train0.ravel())
#print("R2_svr=",svr.score(X_train0, y_train0.ravel()))

predictions_svr = svr.predict(X_test0)
plt.figure()
plt.scatter(X_train0, y_train0)
plt.scatter(X_test0, y_test0)
plt.scatter(X_test0, predictions_svr, c='r')
plt.title('SVR')


#DecisionTree
tree = DecisionTreeRegressor(max_depth=10, min_samples_split=10)
tree.fit(X_train0, y_train0)
#print("R2_tree=",tree.score(X_train0, y_train0))

predictions_tree = tree.predict(X_test0)
plt.figure()
plt.scatter(X_train0, y_train0)
plt.scatter(X_test0, y_test0)
plt.scatter(X_test0, predictions_tree, c='r')
plt.title('tree')


#RandomForest
forest = RandomForestRegressor()
forest.fit(X_train0, y_train0.ravel())
#print("R2_forest=",forest.score(X_train0, y_train0.ravel()))

predictions_forest = forest.predict(X_test0)
plt.figure()
plt.scatter(X_train0, y_train0)
plt.scatter(X_test0, y_test0)
plt.scatter(X_test0, predictions_forest, c='r')
plt.title('forest')


####CLASSIFICATION
#KNN
knnc = KNeighborsClassifier(3)
knnc.fit(X_trainA, y_trainA.ravel())
#print("R2_knnc=",knnc.score(X_trainA, y_trainA.ravel()))

predictions_knnc = knnc.predict(X_testA)
"""plt.figure()
plt.scatter(X_trainA['date'], y_trainA)
plt.scatter(X_testA['date'], y_testA)
plt.scatter(X_testA['date'], predictions_knnc, c='r')
plt.title('KNN date')"""
plt.figure()
plt.scatter(X_trainA['waste'], y_trainA)
plt.scatter(X_testA['waste'], y_testA)
plt.scatter(X_testA['waste'], predictions_knnc, c='r')
plt.title('KNN #waste')


###SVC (kernel='rbf')
svc = SVC(C=100)
svc.fit(X_trainA, y_trainA.ravel())
#print("R2_svc=",svc.score(X_trainA, y_trainA.ravel()))

predictions_svc = svc.predict(X_testA)
plt.figure()
plt.scatter(X_trainA['waste'], y_trainA)
plt.scatter(X_testA['waste'], y_testA)
plt.scatter(X_testA['waste'], predictions_svc, c='r')
plt.title('svc')


#DecisionTree
treec = DecisionTreeClassifier(max_depth=10, min_samples_split=10)
treec.fit(X_trainA, y_trainA.ravel())
#print("R2_treec=",treec.score(X_trainA, y_trainA.ravel()))

predictions_treec = treec.predict(X_testA)
plt.figure()
plt.scatter(X_trainA['waste'], y_trainA)
plt.scatter(X_testA['waste'], y_testA)
plt.scatter(X_testA['waste'], predictions_treec, c='r')
plt.title('treec')


#RandomForest
forestc = RandomForestClassifier()
forestc.fit(X_trainA, y_trainA.ravel())
#print("R2_forestc=",forestc.score(X_trainA, y_trainA.ravel()))

predictions_forestc = forestc.predict(X_testA)
plt.figure()
plt.scatter(X_trainA['waste'], y_trainA)
plt.scatter(X_testA['waste'], y_testA)
plt.scatter(X_testA['waste'], predictions_forestc, c='r')
plt.title('forestc')



"""Optimization"""
###CrossValidation
cv = ShuffleSplit(3, test_size=0.2) 
#print('Mean CV_lasso =', cross_val_score(lasso, X_train0, y_train0.ravel(), cv=cv).mean())
#print('Mean CV_en =', cross_val_score(elasticnet, X_train0, y_train0.ravel(), cv=cv).mean())
# print('Mean CV_knn =', cross_val_score(knn, X_train0, y_train0.ravel(), cv=cv).mean())       #.ravel to avoid DataConversionWarning: 'A column-vector y was passed when a 1d array was expected'
# print('Mean CV_svr =', cross_val_score(svr, X_train0, y_train0.ravel(), cv=cv).mean())
# print('Mean CV_tree =', cross_val_score(tree, X_train0, y_train0.ravel(), cv=cv).mean())
# print('Mean CV_forest =', cross_val_score(forest, X_train0, y_train0.ravel(), cv=cv).mean())

# print('Mean CV_knnc =', cross_val_score(knnc, X_trainA, y_trainA.ravel(), cv=cv).mean())
# print('Mean CV_svc =', cross_val_score(svc, X_trainA, y_trainA.ravel(), cv=cv).mean())
# print('Mean CV_treec =', cross_val_score(treec, X_trainA, y_trainA.ravel(), cv=cv).mean())
# print('Mean CV_forestc =', cross_val_score(forestc, X_trainA, y_trainA.ravel(), cv=cv).mean())


###Optimisation of hyperparameters (Gridsearchcv)
def construct_model_reg(X_train, y_train):
    #regression_models = [linear_model.Lasso(), linear_model.ElasticNet(), KNeighborsRegressor(), SVR()]
    #lasso_parameters = {'alpha': np.arange(0.1, 1.0, 0.1)}
    #en_parameters = {'alpha': np.arange(0.1, 1.0, 0.1), 'l1_ratio': np.arange(0.1, 1.0, 0.1)}
    regression_models = [KNeighborsRegressor(), SVR(), DecisionTreeRegressor(), RandomForestRegressor()]  # the list of regression classifiers to use
    knn_parameters = {'n_neighbors': np.arange(1, 15, 1), 'weights':['uniform', 'distance'], 'p':[1, 2, 3]}
    svr_parameters = {'C': np.arange(50, 200, 10), 'kernel':['poly', 'rbf', 'sigmoid'], 'epsilon':np.arange(0.1, 0.5, 0.1)}
    tree_parameters = {'criterion':['squared_error', 'friedman_mse', 'absolute_error'], 'max_depth': np.arange(1, 10, 1)}  #'random_state':[123]}
    forest_parameters = {'n_estimators': np.arange(10, 100, 10), 'criterion':['squared_error', 'friedman_mse', 'absolute_error']} #'random_state':[123]}
        #criterion: not 'poisson' because we have negative values du to te scaling
    parameters = [knn_parameters, svr_parameters, tree_parameters, forest_parameters]
    #parameters = [lasso_parameters, en_parameters, knn_parameters, svr_parameters]
    estimators = []
        
    # iterate through each classifier and use GridSearchCV
    for i,model in enumerate(regression_models):
        grid = GridSearchCV(model,
                            param_grid = parameters[i],
                            cv=3)
        grid.fit(X_train, y_train)
        estimators.append([grid.best_params_, grid.best_score_, grid.best_estimator_])
    
    return estimators


def construct_model_clas(X_train, y_train):
    regression_models = [KNeighborsClassifier(), SVC(), DecisionTreeClassifier(), RandomForestClassifier()]  # the list of classification classifiers to use
    knnc_parameters = {'n_neighbors': np.arange(1, 15, 1), 'weights':['uniform', 'distance'], 'p':[1, 2, 3]}
    svc_parameters = {'C': np.arange(50, 200, 10), 'kernel':['poly', 'rbf', 'sigmoid']} 
    treec_parameters = {'criterion':['gini', 'entropy'],'max_depth': np.arange(1, 10, 1)}
    forestc_parameters = {'n_estimators': np.arange(10, 100, 10), 'criterion':['gini', 'entropy']}
        #criterion: not 'poisson' because we have negative values du to te scaling
    parameters = [knnc_parameters, svc_parameters, treec_parameters, forestc_parameters]
    estimators = []
        
    # iterate through each classifier and use GridSearchCV
    for i,model in enumerate(regression_models):
        grid = GridSearchCV(model,
                            param_grid = parameters[i],
                            cv=3)   
        grid.fit(X_train, y_train)
        estimators.append([grid.best_params_, grid.best_score_, grid.best_estimator_])
    
    return estimators


estim_reg = construct_model_reg(X_train0, y_train0.ravel())
estim_clas = construct_model_clas(X_trainA, y_trainA.ravel())
print(estim_reg)
print(estim_clas)


###Saving the new optimized models
knn_op = estim_reg[0][2]
svr_op = estim_reg[1][2]
tree_op = estim_reg[2][2]
forest_op = estim_reg[3][2]

knn_op.fit(X_train0, y_train0)
predictions_knn_op = knn_op.predict(X_test0)
svr_op.fit(X_train0, y_train0.ravel())
predictions_svr_op = svr_op.predict(X_test0)
tree_op.fit(X_train0, y_train0.ravel())
predictions_tree_op = tree_op.predict(X_test0)
forest_op.fit(X_train0, y_train0.ravel())
predictions_forest_op = forest_op.predict(X_test0)


knnc_op = estim_clas[0][2]
svc_op = estim_clas[1][2]
treec_op = estim_clas[2][2]
forestc_op = estim_clas[3][2]

knnc_op.fit(X_trainA, y_trainA.ravel())
predictions_knnc_op = knnc_op.predict(X_testA)
svc_op.fit(X_trainA, y_trainA.ravel())
predictions_svc_op = svc_op.predict(X_testA)
treec_op.fit(X_trainA, y_trainA.ravel())
predictions_treec_op = treec_op.predict(X_testA)
forestc_op.fit(X_trainA, y_trainA.ravel())
predictions_forestc_op = forestc_op.predict(X_testA)


"""RESULTS"""

###Table of results REG
head = np.array([['state', 'metrics', 'KNN', 'SVR', 'DTR', 'RFR']])
#print(head.shape)

C1 = np.array([[i] for i in ['non-optimized', 'grid'] for j in range(3)])
#print(C1.shape)
C2 = np.array([[i] for j in range(2) for i in ['R2','MAE', 'MSE']])
#print(C2.shape)
C1_2 = np.concatenate((C1, C2), axis=1)
#print(C1_2.shape)

results_reg = [[knn.score(X_train0, y_train0), svr.score(X_train0, y_train0), tree.score(X_train0, y_train0), forest.score(X_train0, y_train0)], 
               [mean_absolute_error(y_test0, i) for i in [predictions_knn, predictions_svr, predictions_tree, predictions_forest]],
               [mean_squared_error(y_test0, i) for i in [predictions_knn, predictions_svr, predictions_tree, predictions_forest]],
               [estim_reg[i][1] for i in range(4)],
               [mean_absolute_error(y_test0, i) for i in [predictions_knn_op, predictions_svr_op, predictions_tree_op, predictions_forest_op]],
               [mean_squared_error(y_test0, i) for i in [predictions_knn_op, predictions_svr_op, predictions_tree_op, predictions_forest_op]]]
           
results_reg = np.array(results_reg)
results_reg = np.round(results_reg, 4)
#print(results.shape)

results_reg = np.concatenate((C1_2, results_reg), axis=1)
results_reg = np.concatenate((head, results_reg), axis=0)
results_reg_df = pd.DataFrame(results_reg)
print(results_reg_df)


###Table of results CLAS
head_c = np.array([['state', 'metrics', 'KNN', 'SVC', 'DTC', 'RFC']])
#print(head.shape)

C1 = np.array([[i] for i in ['non-optimized', 'grid'] for j in range(2)])
#print(C1.shape)
C2 = np.array([[i] for j in range(2) for i in ['R2','accuracy']])
#print(C2.shape)
C1_2 = np.concatenate((C1, C2), axis=1)
#print(C1_2.shape)
results_clas = [[knnc.score(X_trainA, y_trainA.ravel()), svc.score(X_trainA, y_trainA.ravel()), treec.score(X_trainA, y_trainA.ravel()), forestc.score(X_trainA, y_trainA.ravel())], 
               [accuracy_score(y_testA, i) for i in [predictions_knnc, predictions_svc, predictions_treec, predictions_forestc]],
               [estim_clas[i][1] for i in range(4)],
               [accuracy_score(y_testA, i) for i in [predictions_knnc_op, predictions_svc_op, predictions_treec_op, predictions_forestc_op]]]
               
results_clas = np.array(results_clas)
results_clas = np.round(results_clas, 4)
#print(results.shape)

results_clas = np.concatenate((C1_2, results_clas), axis=1)
results_clas = np.concatenate((head, results_clas), axis=0)
results_clas_df = pd.DataFrame(results_clas)
print(results_clas_df)


###saving the table of results in excel
excel = pd.ExcelWriter('resultsV3.2.2.xlsx')
results_reg_df.to_excel(excel, sheet_name='REG', index=False)
results_clas_df.to_excel(excel, sheet_name='CLAS', index=False)
excel.save()


###graf optimized models
#reg
plt.figure(figsize = (15, 10))
plt.gcf().subplots_adjust(left = 0.3, bottom = 0.3,
                       right = 0.7, top = 0.7, wspace = 0.2, hspace = 0.7)
plt.subplot(2,2,1)
plt.scatter(X_train0, y_train0)
plt.scatter(X_test0, y_test0)
plt.scatter(X_test0, predictions_knn_op, c='r')
plt.xlabel('date')
plt.ylabel('waste generation (in tons)')
plt.title('KNN')
plt.subplot(2,2,2)
plt.scatter(X_train0, y_train0)
plt.scatter(X_test0, y_test0)
plt.scatter(X_test0, predictions_svr_op, c='r')
plt.xlabel('date')
plt.ylabel('waste generation (in tons)')
plt.title('SVR')
plt.subplot(2,2,3)
plt.scatter(X_train0, y_train0)
plt.scatter(X_test0, y_test0)
plt.scatter(X_test0, predictions_tree_op, c='r')
plt.xlabel('date')
plt.ylabel('waste generation (in tons)')
plt.title('Decision tree')
plt.subplot(2,2,4)
plt.scatter(X_train0, y_train0)
plt.scatter(X_test0, y_test0)
plt.scatter(X_test0, predictions_forest_op, c='r')
plt.xlabel('date')
plt.ylabel('waste generation (in tons)')
plt.title('Random forest')

#clas
plt.figure(figsize = (15, 10))
plt.gcf().subplots_adjust(left = 0.3, bottom = 0.3,
                       right = 0.7, top = 0.7, wspace = 0.2, hspace = 0.7)
plt.subplot(2,2,1)
plt.scatter(X_trainA['waste'], y_trainA)
plt.scatter(X_testA['waste'], y_testA)
plt.scatter(X_testA['waste'], predictions_knnc_op, c='r')
plt.ylabel('categories')
plt.xlabel('waste generation (in tons)')
plt.title('KNN')
plt.subplot(2,2,2)
plt.scatter(X_trainA['waste'], y_trainA)
plt.scatter(X_testA['waste'], y_testA)
plt.scatter(X_testA['waste'], predictions_svc_op, c='r')
plt.ylabel('categories')
plt.xlabel('waste generation (in tons)')
plt.title('SVC')
plt.subplot(2,2,3)
plt.scatter(X_trainA['waste'], y_trainA)
plt.scatter(X_testA['waste'], y_testA)
plt.scatter(X_testA['waste'], predictions_treec_op, c='r')
plt.ylabel('categories')
plt.xlabel('waste generation (in tons)')
plt.title('Decision Tree')
plt.subplot(2,2,4)
plt.scatter(X_trainA['waste'], y_trainA)
plt.scatter(X_testA['waste'], y_testA)
plt.scatter(X_testA['waste'], predictions_forestc_op, c='r')
plt.ylabel('categories')
plt.xlabel('waste generation (in tons)')
plt.title('Random Forest')


"""
###Learning curves
#N_lasso, train_score_lasso, val_score_lasso = learning_curve(lasso, X_train0, y_train0, train_sizes=np.linspace(0.1, 1.0, 10), cv=5)
#N_en, train_score_en, val_score_en = learning_curve(elasticnet, X_train0, y_train0, train_sizes=np.linspace(0.1, 1.0, 10), cv=5)
N_knn, train_score_knn, val_score_knn = learning_curve(knn, X_train0, y_train0.ravel(), train_sizes=np.linspace(0.5, 1.0, 5), cv=2)
N_svr, train_score_svr, val_score_svr = learning_curve(svr, X_train0, y_train0.ravel(), train_sizes=np.linspace(0.5, 1.0, 5), cv=2)
N_tree, train_score_tree, val_score_tree = learning_curve(tree, X_train0, y_train0.ravel(), train_sizes=np.linspace(0.5, 1.0, 5), cv=2)
N_forest, train_score_forest, val_score_forest = learning_curve(forest, X_train0, y_train0.ravel(), train_sizes=np.linspace(0.5, 1.0, 5), cv=2)

plt.figure()
plt.subplot(2,2,1)
plt.plot(N_knn,train_score_knn.mean(axis=1), label='knn')
plt.legend()
plt.title('Training curves')
plt.subplot(2,2,2)
plt.plot(N_svr,train_score_svr.mean(axis=1), label='svr')
plt.legend()
plt.subplot(2,2,3)
plt.plot(N_tree,train_score_tree.mean(axis=1), label='tree')
plt.legend()
plt.subplot(2,2,4)
plt.plot(N_forest,train_score_forest.mean(axis=1), label='forest')
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
plt.plot(N_forest,val_score_forest.mean(axis=1), label='forest')
plt.legend()


N_knnc, train_score_knnc, val_score_knnc = learning_curve(knnc, X_trainA, y_trainA.ravel(), train_sizes=np.linspace(0.5, 1.0, 5), cv=2)
N_svc, train_score_svc, val_score_svc = learning_curve(svc, X_trainA, y_trainA.ravel(), train_sizes=np.linspace(0.5, 1.0, 5), cv=2)
N_treec, train_score_treec, val_score_treec = learning_curve(treec, X_trainA, y_trainA.ravel(), train_sizes=np.linspace(0.5, 1.0, 5), cv=2)
N_forestc, train_score_forestc, val_score_forestc = learning_curve(forestc, X_trainA, y_trainA.ravel(), train_sizes=np.linspace(0.5, 1.0, 5), cv=2)

plt.figure()
plt.subplot(2,2,1)
plt.plot(N_knnc,train_score_knnc.mean(axis=1), label='knnc')
plt.legend()
plt.title('Training curves')
plt.subplot(2,2,2)
plt.plot(N_svc,train_score_svc.mean(axis=1), label='svc')
plt.legend()
plt.subplot(2,2,3)
plt.plot(N_treec,train_score_treec.mean(axis=1), label='treec')
plt.legend()
plt.subplot(2,2,4)
plt.plot(N_forestc,train_score_forestc.mean(axis=1), label='forestc')
plt.legend()

plt.figure()
plt.subplot(2,2,1)
plt.plot(N_knnc,val_score_knnc.mean(axis=1), label='knnc')
plt.legend()
plt.title('Validation curves')
plt.subplot(2,2,2)
plt.plot(N_svc,val_score_svc.mean(axis=1), label='svc')
plt.legend()
plt.subplot(2,2,3)
plt.plot(N_treec,val_score_treec.mean(axis=1), label='treec')
plt.legend()
plt.subplot(2,2,4)
plt.plot(N_forestc,val_score_forestc.mean(axis=1), label='forestc')
plt.legend()
"""