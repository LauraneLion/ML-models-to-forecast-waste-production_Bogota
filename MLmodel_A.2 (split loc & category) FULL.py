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

np.random.seed(0)  # fix the random state


"""PreProcessing"""
# Uploading of the Dataset
data = pd.read_excel('Dataset21_24.xlsx')
# print(data.head())

y = data['waste generation'].values
X = data[['date', 'area', 'category of waste']].values

y = y.reshape((y.shape[0], 1))  # default value : (1129,)
#print(y.shape, X.shape)


# Encoding
encoder = OrdinalEncoder()
X = encoder.fit_transform(X)
# print(X)


# Scaling
scaler = RobustScaler()
y = scaler.fit_transform(y)
# print(y)


# Creation of the dataframes for each area and category
# conversion in dataframe because fit_transform return an array
X_df = pd.DataFrame(X)
y_df = pd.DataFrame(y)
# the same as data, but with encoding and scaling
df = pd.concat([y_df, X_df], axis=1)
df.columns = ['waste', 'date', 'area', 'category']
# print(df.head())

ASE0_0 = df[(df['area'] == 0) & (df['category'] == 0)]
ASE0_1 = df[(df['area'] == 0) & (df['category'] == 1)]
ASE0_2 = df[(df['area'] == 0) & (df['category'] == 2)]
ASE0_3 = df[(df['area'] == 0) & (df['category'] == 3)]
ASE0_4 = df[(df['area'] == 0) & (df['category'] == 4)]
ASE0_5 = df[(df['area'] == 0) & (df['category'] == 5)]
ASE0_6 = df[(df['area'] == 0) & (df['category'] == 6)]
ASE0_7 = df[(df['area'] == 0) & (df['category'] == 7)]

ASE1_0 = df[(df['area'] == 1) & (df['category'] == 0)]
ASE1_1 = df[(df['area'] == 1) & (df['category'] == 1)]
ASE1_2 = df[(df['area'] == 1) & (df['category'] == 2)]
ASE1_3 = df[(df['area'] == 1) & (df['category'] == 3)]
ASE1_4 = df[(df['area'] == 1) & (df['category'] == 4)]
ASE1_5 = df[(df['area'] == 1) & (df['category'] == 5)]
ASE1_6 = df[(df['area'] == 1) & (df['category'] == 6)]
ASE1_7 = df[(df['area'] == 1) & (df['category'] == 7)]

ASE2_0 = df[(df['area'] == 2) & (df['category'] == 0)]
ASE2_1 = df[(df['area'] == 2) & (df['category'] == 1)]
ASE2_2 = df[(df['area'] == 2) & (df['category'] == 2)]
ASE2_3 = df[(df['area'] == 2) & (df['category'] == 3)]
ASE2_4 = df[(df['area'] == 2) & (df['category'] == 4)]
ASE2_5 = df[(df['area'] == 2) & (df['category'] == 5)]
ASE2_6 = df[(df['area'] == 2) & (df['category'] == 6)]
ASE2_7 = df[(df['area'] == 2) & (df['category'] == 7)]

ASE3_0 = df[(df['area'] == 3) & (df['category'] == 0)]
ASE3_1 = df[(df['area'] == 3) & (df['category'] == 1)]
ASE3_2 = df[(df['area'] == 3) & (df['category'] == 2)]
ASE3_3 = df[(df['area'] == 3) & (df['category'] == 3)]
ASE3_4 = df[(df['area'] == 3) & (df['category'] == 4)]
ASE3_5 = df[(df['area'] == 3) & (df['category'] == 5)]
ASE3_6 = df[(df['area'] == 3) & (df['category'] == 6)]
ASE3_7 = df[(df['area'] == 3) & (df['category'] == 7)]

ASE4_0 = df[(df['area'] == 4) & (df['category'] == 0)]
ASE4_1 = df[(df['area'] == 4) & (df['category'] == 1)]
ASE4_2 = df[(df['area'] == 4) & (df['category'] == 2)]
ASE4_3 = df[(df['area'] == 4) & (df['category'] == 3)]
ASE4_4 = df[(df['area'] == 4) & (df['category'] == 4)]
ASE4_5 = df[(df['area'] == 4) & (df['category'] == 5)]
ASE4_6 = df[(df['area'] == 4) & (df['category'] == 6)]
ASE4_7 = df[(df['area'] == 4) & (df['category'] == 7)]


# Creation of Trainset and Testset
X_train00, X_test00, y_train00, y_test00 = train_test_split(
    ASE0_0['date'], ASE0_0['waste'], test_size=0.2)
X_train01, X_test01, y_train01, y_test01 = train_test_split(
    ASE0_1['date'], ASE0_1['waste'], test_size=0.2)
X_train02, X_test02, y_train02, y_test02 = train_test_split(
    ASE0_2['date'], ASE0_2['waste'], test_size=0.2)
X_train03, X_test03, y_train03, y_test03 = train_test_split(
    ASE0_3['date'], ASE0_3['waste'], test_size=0.2)
X_train04, X_test04, y_train04, y_test04 = train_test_split(
    ASE0_4['date'], ASE0_4['waste'], test_size=0.2)
X_train05, X_test05, y_train05, y_test05 = train_test_split(
    ASE0_5['date'], ASE0_5['waste'], test_size=0.2)
X_train06, X_test06, y_train06, y_test06 = train_test_split(
    ASE0_6['date'], ASE0_6['waste'], test_size=0.2)
X_train07, X_test07, y_train07, y_test07 = train_test_split(
    ASE0_7['date'], ASE0_7['waste'], test_size=0.2)

X_train10, X_test10, y_train10, y_test10 = train_test_split(
    ASE1_0['date'], ASE1_0['waste'], test_size=0.2)
X_train11, X_test11, y_train11, y_test11 = train_test_split(
    ASE1_1['date'], ASE1_1['waste'], test_size=0.2)
X_train12, X_test12, y_train12, y_test12 = train_test_split(
    ASE1_2['date'], ASE1_2['waste'], test_size=0.2)
X_train13, X_test13, y_train13, y_test13 = train_test_split(
    ASE1_3['date'], ASE1_3['waste'], test_size=0.2)
X_train14, X_test14, y_train14, y_test14 = train_test_split(
    ASE1_4['date'], ASE1_4['waste'], test_size=0.2)
X_train15, X_test15, y_train15, y_test15 = train_test_split(
    ASE1_5['date'], ASE1_5['waste'], test_size=0.2)
X_train16, X_test16, y_train16, y_test16 = train_test_split(
    ASE1_6['date'], ASE1_6['waste'], test_size=0.2)
X_train17, X_test17, y_train17, y_test17 = train_test_split(
    ASE1_7['date'], ASE1_7['waste'], test_size=0.2)

X_train20, X_test20, y_train20, y_test20 = train_test_split(
    ASE2_0['date'], ASE2_0['waste'], test_size=0.2)
X_train21, X_test21, y_train21, y_test21 = train_test_split(
    ASE2_1['date'], ASE2_1['waste'], test_size=0.2)
X_train22, X_test22, y_train22, y_test22 = train_test_split(
    ASE2_2['date'], ASE2_2['waste'], test_size=0.2)
X_train23, X_test23, y_train23, y_test23 = train_test_split(
    ASE2_3['date'], ASE2_3['waste'], test_size=0.2)
X_train24, X_test24, y_train24, y_test24 = train_test_split(
    ASE2_4['date'], ASE2_4['waste'], test_size=0.2)
X_train25, X_test25, y_train25, y_test25 = train_test_split(
    ASE2_5['date'], ASE2_5['waste'], test_size=0.2)
X_train26, X_test26, y_train26, y_test26 = train_test_split(
    ASE2_6['date'], ASE2_6['waste'], test_size=0.2)
X_train27, X_test27, y_train27, y_test27 = train_test_split(
    ASE2_7['date'], ASE2_7['waste'], test_size=0.2)

X_train30, X_test30, y_train30, y_test30 = train_test_split(
    ASE3_0['date'], ASE3_0['waste'], test_size=0.2)
X_train31, X_test31, y_train31, y_test31 = train_test_split(
    ASE3_1['date'], ASE3_1['waste'], test_size=0.2)
X_train32, X_test32, y_train32, y_test32 = train_test_split(
    ASE3_2['date'], ASE3_2['waste'], test_size=0.2)
X_train33, X_test33, y_train33, y_test33 = train_test_split(
    ASE3_3['date'], ASE3_3['waste'], test_size=0.2)
X_train34, X_test34, y_train34, y_test34 = train_test_split(
    ASE3_4['date'], ASE3_4['waste'], test_size=0.2)
X_train35, X_test35, y_train35, y_test35 = train_test_split(
    ASE3_5['date'], ASE3_5['waste'], test_size=0.2)
X_train36, X_test36, y_train36, y_test36 = train_test_split(
    ASE3_6['date'], ASE3_6['waste'], test_size=0.2)
X_train37, X_test37, y_train37, y_test37 = train_test_split(
    ASE3_7['date'], ASE3_7['waste'], test_size=0.2)

X_train40, X_test40, y_train40, y_test40 = train_test_split(
    ASE4_0['date'], ASE4_0['waste'], test_size=0.2)
X_train41, X_test41, y_train41, y_test41 = train_test_split(
    ASE4_1['date'], ASE4_1['waste'], test_size=0.2)
X_train42, X_test42, y_train42, y_test42 = train_test_split(
    ASE4_2['date'], ASE4_2['waste'], test_size=0.2)
X_train43, X_test43, y_train43, y_test43 = train_test_split(
    ASE4_3['date'], ASE4_3['waste'], test_size=0.2)
X_train44, X_test44, y_train44, y_test44 = train_test_split(
    ASE4_4['date'], ASE4_4['waste'], test_size=0.2)
X_train45, X_test45, y_train45, y_test45 = train_test_split(
    ASE4_5['date'], ASE4_5['waste'], test_size=0.2)
X_train46, X_test46, y_train46, y_test46 = train_test_split(
    ASE4_6['date'], ASE4_6['waste'], test_size=0.2)
X_train47, X_test47, y_train47, y_test47 = train_test_split(
    ASE4_7['date'], ASE4_7['waste'], test_size=0.2)

#print(X_train00.shape, X_test00.shape, y_train00.shape, y_test00.shape)
# Reshape because all the train and test set have shape = (n,)
X_train02 = X_train02.values.reshape((X_train02.shape[0], 1))
y_train02 = y_train02.values.reshape((X_train02.shape[0], 1))
X_test02 = X_test02.values.reshape((X_test02.shape[0], 1))
y_test02 = y_test02.values.reshape((X_test02.shape[0], 1))

X_train12 = X_train12.values.reshape((X_train12.shape[0], 1))
y_train12 = y_train12.values.reshape((y_train12.shape[0], 1))
X_test12 = X_test12.values.reshape((X_test12.shape[0], 1))
y_test12 = y_test12.values.reshape((y_test12.shape[0], 1))

X_train22 = X_train22.values.reshape((X_train22.shape[0], 1))
y_train22 = y_train22.values.reshape((y_train22.shape[0], 1))
X_test22 = X_test22.values.reshape((X_test22.shape[0], 1))
y_test22 = y_test22.values.reshape((y_test22.shape[0], 1))

X_train32 = X_train32.values.reshape((X_train32.shape[0], 1))
y_train32 = y_train32.values.reshape((y_train32.shape[0], 1))
X_test32 = X_test32.values.reshape((X_test32.shape[0], 1))
y_test32 = y_test32.values.reshape((y_test32.shape[0], 1))

X_train42 = X_train42.values.reshape((X_train42.shape[0], 1))
y_train42 = y_train42.values.reshape((y_train42.shape[0], 1))
X_test42 = X_test42.values.reshape((X_test42.shape[0], 1))
y_test42 = y_test42.values.reshape((y_test42.shape[0], 1))

"""plt.figure()
plt.scatter(X_train02, y_train02)
plt.scatter(X_test02, y_test02)
plt.figure()
plt.scatter(X_train12, y_train12)
plt.scatter(X_test12, y_test12)
plt.figure()
plt.scatter(X_train22, y_train22)
plt.scatter(X_test22, y_test22)
plt.figure()
plt.scatter(X_train32, y_train32)
plt.scatter(X_test32, y_test32)
plt.figure()
plt.scatter(X_train42, y_train42)
plt.scatter(X_test42, y_test42)"""



"""Optimized Modelling"""
### Optimisation of hyperparameters
# https://www.stochasticbard.com/blog/lasso_regression/
def construct_model(X_train, y_train):
    regression_models = [KNeighborsRegressor(), SVR(), DecisionTreeRegressor(), RandomForestRegressor()]  # the list of classifiers to use
    knn_parameters = {'n_neighbors': np.arange(1, 15, 1), 'weights': ['uniform', 'distance'], 'p': [1, 2, 3]}
    svr_parameters = {'C': np.arange(50, 200, 10), 'kernel': ['poly', 'rbf', 'sigmoid'], 'epsilon': np.arange(0.1, 0.5, 0.1)}
    tree_parameters = {'criterion': ['squared_error', 'friedman_mse', 'absolute_error'], 'max_depth': np.arange(1, 10, 1)}  # 'random_state':[123]}
    forest_parameters = {'n_estimators': np.arange(10, 100, 10), 'criterion': ['squared_error', 'friedman_mse', 'absolute_error']}  # 'random_state':[123]}
    # criterion: not 'poisson' because we have negative values du to te scaling
    parameters = [knn_parameters, svr_parameters, tree_parameters, forest_parameters]
    estimators = []

    # iterate through each classifier and use GridSearchCV
    for i, model in enumerate(regression_models):
        grid = GridSearchCV(model,
                            param_grid=parameters[i],  # hyperparameters
                            cv=3)
        grid.fit(X_train, y_train)
        estimators.append([grid.best_params_, grid.best_score_, grid.best_estimator_])
    return estimators



"""ALL ASE, HOUSEHOLD WASTE"""

estim02 = construct_model(X_train02, y_train02.ravel())
estim12 = construct_model(X_train12, y_train12.ravel())
estim22 = construct_model(X_train22, y_train22.ravel())
estim32 = construct_model(X_train32, y_train32.ravel())
estim42 = construct_model(X_train42, y_train42.ravel())

#optimization of polynomial hyperparameters
poly = LinearRegression()
R2_poly02 = []
MAE_poly02 = []
MSE_poly02 = []
for i in range(1, 16):
    degree = PolynomialFeatures(degree=i, include_bias=False)
    poly_features = degree.fit_transform(X_train02)
    poly_test = degree.fit_transform(X_test02)
    poly.fit(poly_features, y_train02)
    R2_poly02.append(poly.score(poly_features, y_train02))
    predictions_poly02 = poly.predict(poly_test)
    MAE_poly02.append(mean_absolute_error(y_test02, predictions_poly02))
    MSE_poly02.append(mean_squared_error(y_test02, predictions_poly02))

R2_poly12 = []
MAE_poly12 = []
MSE_poly12 = []
for i in range(1, 16):
    degree = PolynomialFeatures(degree=i, include_bias=False)
    poly_features = degree.fit_transform(X_train12)
    poly_test = degree.fit_transform(X_test12)
    poly.fit(poly_features, y_train12)
    R2_poly12.append(poly.score(poly_features, y_train12))
    predictions_poly12 = poly.predict(poly_test)
    MAE_poly12.append(mean_absolute_error(y_test12, predictions_poly12))
    MSE_poly12.append(mean_squared_error(y_test12, predictions_poly12))

R2_poly22 = []
MAE_poly22 = []
MSE_poly22 = []
for i in range(1, 16):
    degree = PolynomialFeatures(degree=i, include_bias=False)
    poly_features = degree.fit_transform(X_train22)
    poly_test = degree.fit_transform(X_test22)
    poly.fit(poly_features, y_train22)
    R2_poly22.append(poly.score(poly_features, y_train22))
    predictions_poly22 = poly.predict(poly_test)
    MAE_poly22.append(mean_absolute_error(y_test22, predictions_poly22))
    MSE_poly22.append(mean_squared_error(y_test22, predictions_poly22))

R2_poly32 = []
MAE_poly32 = []
MSE_poly32 = []
for i in range(1, 16):
    degree = PolynomialFeatures(degree=i, include_bias=False)
    poly_features = degree.fit_transform(X_train32)
    poly_test = degree.fit_transform(X_test32)
    poly.fit(poly_features, y_train32)
    R2_poly32.append(poly.score(poly_features, y_train32))
    predictions_poly32 = poly.predict(poly_test)
    MAE_poly32.append(mean_absolute_error(y_test32, predictions_poly32))
    MSE_poly32.append(mean_squared_error(y_test32, predictions_poly32))

R2_poly42 = []
MAE_poly42 = []
MSE_poly42 = []
for i in range(1, 16):
    degree = PolynomialFeatures(degree=i, include_bias=False)
    poly_features = degree.fit_transform(X_train42)
    poly_test = degree.fit_transform(X_test42)
    poly.fit(poly_features, y_train42)
    R2_poly42.append(poly.score(poly_features, y_train42))
    predictions_poly42 = poly.predict(poly_test)
    MAE_poly42.append(mean_absolute_error(y_test42, predictions_poly42))
    MSE_poly42.append(mean_squared_error(y_test42, predictions_poly42))


### Saving the new optimized models
degree_op02 = PolynomialFeatures(degree=(R2_poly02.index(max(R2_poly02)))+1, include_bias=False)
poly_op02 = LinearRegression()
knn_op02 = estim02[0][2]
svr_op02 = estim02[1][2]
tree_op02 = estim02[2][2]
forest_op02 = estim02[3][2]
poly_features_op02 = degree_op02.fit_transform(X_train02)
poly_test_op02 = degree_op02.fit_transform(X_test02)
poly_op02.fit(poly_features_op02, y_train02)
predictions_poly_op02 = poly_op02.predict(poly_test_op02)
knn_op02.fit(X_train02, y_train02)
predictions_knn_op02 = knn_op02.predict(X_test02)
svr_op02.fit(X_train02, y_train02.ravel())
predictions_svr_op02 = svr_op02.predict(X_test02)
tree_op02.fit(X_train02, y_train02.ravel())
predictions_tree_op02 = tree_op02.predict(X_test02)
forest_op02.fit(X_train02, y_train02.ravel())
predictions_forest_op02 = forest_op02.predict(X_test02)

degree_op12 = PolynomialFeatures(degree=(R2_poly12.index(max(R2_poly12)))+1, include_bias=False)
poly_op12 = LinearRegression()
knn_op12 = estim12[0][2]
svr_op12 = estim12[1][2]
tree_op12 = estim12[2][2]
forest_op12 = estim12[3][2]
poly_features_op12 = degree_op12.fit_transform(X_train12)
poly_test_op12 = degree_op12.fit_transform(X_test12)
poly_op12.fit(poly_features_op12, y_train12)
predictions_poly_op12 = poly_op12.predict(poly_test_op12)
knn_op12.fit(X_train12, y_train12)
predictions_knn_op12 = knn_op12.predict(X_test12)
svr_op12.fit(X_train12, y_train12.ravel())
predictions_svr_op12 = svr_op12.predict(X_test12)
tree_op12.fit(X_train12, y_train12.ravel())
predictions_tree_op12 = tree_op12.predict(X_test12)
forest_op12.fit(X_train12, y_train12.ravel())
predictions_forest_op12 = forest_op12.predict(X_test12)

degree_op22 = PolynomialFeatures(degree=(R2_poly22.index(max(R2_poly22)))+1, include_bias=False)
poly_op22 = LinearRegression()
knn_op22 = estim22[0][2]
svr_op22 = estim22[1][2]
tree_op22 = estim22[2][2]
forest_op22 = estim22[3][2]
poly_features_op22 = degree_op22.fit_transform(X_train22)
poly_test_op22 = degree_op22.fit_transform(X_test22)
poly_op22.fit(poly_features_op22, y_train22)
predictions_poly_op22 = poly_op22.predict(poly_test_op22)
knn_op22.fit(X_train22, y_train22)
predictions_knn_op22 = knn_op22.predict(X_test22)
svr_op22.fit(X_train22, y_train22.ravel())
predictions_svr_op22 = svr_op22.predict(X_test22)
tree_op22.fit(X_train22, y_train22.ravel())
predictions_tree_op22 = tree_op22.predict(X_test22)
forest_op22.fit(X_train22, y_train22.ravel())
predictions_forest_op22 = forest_op22.predict(X_test22)

degree_op32 = PolynomialFeatures(degree=(R2_poly32.index(max(R2_poly32)))+1, include_bias=False)
poly_op32 = LinearRegression()
knn_op32 = estim32[0][2]
svr_op32 = estim32[1][2]
tree_op32 = estim32[2][2]
forest_op32 = estim32[3][2]
poly_features_op32 = degree_op32.fit_transform(X_train32)
poly_test_op32 = degree_op32.fit_transform(X_test32)
poly_op32.fit(poly_features_op32, y_train32)
predictions_poly_op32 = poly_op32.predict(poly_test_op32)
knn_op32.fit(X_train32, y_train32)
predictions_knn_op32 = knn_op32.predict(X_test32)
svr_op32.fit(X_train32, y_train32.ravel())
predictions_svr_op32 = svr_op32.predict(X_test32)
tree_op32.fit(X_train32, y_train32.ravel())
predictions_tree_op32 = tree_op32.predict(X_test32)
forest_op32.fit(X_train32, y_train32.ravel())
predictions_forest_op32 = forest_op32.predict(X_test32)

degree_op42 = PolynomialFeatures(degree=(R2_poly42.index(max(R2_poly42)))+1, include_bias=False)
poly_op42 = LinearRegression()
knn_op42 = estim42[0][2]
svr_op42 = estim42[1][2]
tree_op42 = estim42[2][2]
forest_op42 = estim42[3][2]
poly_features_op42 = degree_op42.fit_transform(X_train42)
poly_test_op42 = degree_op42.fit_transform(X_test42)
poly_op42.fit(poly_features_op42, y_train42)
predictions_poly_op42 = poly_op42.predict(poly_test_op42)
knn_op42.fit(X_train42, y_train42)
predictions_knn_op42 = knn_op42.predict(X_test42)
svr_op42.fit(X_train42, y_train42.ravel())
predictions_svr_op42 = svr_op42.predict(X_test42)
tree_op42.fit(X_train42, y_train42.ravel())
predictions_tree_op42 = tree_op42.predict(X_test42)
forest_op42.fit(X_train42, y_train42.ravel())
predictions_forest_op42 = forest_op42.predict(X_test42)


### Tables of results
head = np.array([['metrics', 'KNN', 'SVR', 'DTR', 'RFR', 'Poly']])
# print(head.shape)
C1 = np.array([['R2'], ['MAE'], ['MSE']])
# print(C1.shape)

results = [[estim02[i][1] for i in range(4)],
           [mean_absolute_error(y_test02, i) for i in [predictions_knn_op02, predictions_svr_op02, predictions_tree_op02, predictions_forest_op02]],
           [mean_squared_error(y_test02, i) for i in [predictions_knn_op02, predictions_svr_op02, predictions_tree_op02, predictions_forest_op02]]]
results = np.array(results)
results = np.round(results, 4)
# print(results.shape)

C_poly = [[np.round(poly_op02.score(poly_features_op02, y_train02), 4)],
          [np.round(mean_absolute_error(y_test02, predictions_poly_op02), 4)],
          [np.round(mean_squared_error(y_test02, predictions_poly_op02), 4)]]
C_poly = np.array(C_poly)

results = np.concatenate((results, C_poly), axis=1)
results = np.concatenate((C1, results), axis=1)
results = np.concatenate((head, results), axis=0)
results_df02 = pd.DataFrame(results)


results = [[estim12[i][1] for i in range(4)],
           [mean_absolute_error(y_test12, i) for i in [predictions_knn_op12, predictions_svr_op12, predictions_tree_op12, predictions_forest_op12]],
           [mean_squared_error(y_test12, i) for i in [predictions_knn_op12, predictions_svr_op12, predictions_tree_op12, predictions_forest_op12]]]
results = np.array(results)
results = np.round(results, 4)
# print(results.shape)

C_poly = [[np.round(poly_op12.score(poly_features_op12, y_train12), 4)],
          [np.round(mean_absolute_error(y_test12, predictions_poly_op12), 4)],
          [np.round(mean_squared_error(y_test12, predictions_poly_op12), 4)]]
C_poly = np.array(C_poly)

results = np.concatenate((results, C_poly), axis=1)
results = np.concatenate((C1, results), axis=1)
results = np.concatenate((head, results), axis=0)
results_df12 = pd.DataFrame(results)


results = [[estim22[i][1] for i in range(4)],
           [mean_absolute_error(y_test22, i) for i in [predictions_knn_op22, predictions_svr_op22, predictions_tree_op22, predictions_forest_op22]],
           [mean_squared_error(y_test12, i) for i in [predictions_knn_op22, predictions_svr_op22, predictions_tree_op22, predictions_forest_op22]]]
results = np.array(results)
results = np.round(results, 4)
# print(results.shape)

C_poly = [[np.round(poly_op22.score(poly_features_op22, y_train22), 4)],
          [np.round(mean_absolute_error(y_test22, predictions_poly_op22), 4)],
          [np.round(mean_squared_error(y_test22, predictions_poly_op22), 4)]]
C_poly = np.array(C_poly)

results = np.concatenate((results, C_poly), axis=1)
results = np.concatenate((C1, results), axis=1)
results = np.concatenate((head, results), axis=0)
results_df22 = pd.DataFrame(results)


results = [[estim32[i][1] for i in range(4)],
           [mean_absolute_error(y_test32, i) for i in [predictions_knn_op32, predictions_svr_op32, predictions_tree_op32, predictions_forest_op32]],
           [mean_squared_error(y_test32, i) for i in [predictions_knn_op32, predictions_svr_op32, predictions_tree_op32, predictions_forest_op32]]]
results = np.array(results)
results = np.round(results, 4)
# print(results.shape)

C_poly = [[np.round(poly_op32.score(poly_features_op32, y_train32), 4)],
          [np.round(mean_absolute_error(y_test32, predictions_poly_op32), 4)],
          [np.round(mean_squared_error(y_test32, predictions_poly_op32), 4)]]
C_poly = np.array(C_poly)

results = np.concatenate((results, C_poly), axis=1)
results = np.concatenate((C1, results), axis=1)
results = np.concatenate((head, results), axis=0)
results_df32 = pd.DataFrame(results)


results = [[estim42[i][1] for i in range(4)],
           [mean_absolute_error(y_test42, i) for i in [predictions_knn_op42, predictions_svr_op42, predictions_tree_op42, predictions_forest_op42]],
           [mean_squared_error(y_test12, i) for i in [predictions_knn_op42, predictions_svr_op42, predictions_tree_op42, predictions_forest_op42]]]
results = np.array(results)
results = np.round(results, 4)
# print(results.shape)

C_poly = [[np.round(poly_op42.score(poly_features_op42, y_train42), 4)],
          [np.round(mean_absolute_error(y_test42, predictions_poly_op42), 4)],
          [np.round(mean_squared_error(y_test42, predictions_poly_op42), 4)]]
C_poly = np.array(C_poly)

results = np.concatenate((results, C_poly), axis=1)
results = np.concatenate((C1, results), axis=1)
results = np.concatenate((head, results), axis=0)
results_df42 = pd.DataFrame(results)


# saving the table of results in excel
excel = pd.ExcelWriter('resultsV2 FULL.xlsx')
results_df02.to_excel(excel, sheet_name='02', index=False)
results_df12.to_excel(excel, sheet_name='12', index=False)
results_df22.to_excel(excel, sheet_name='22', index=False)
results_df32.to_excel(excel, sheet_name='32', index=False)
results_df42.to_excel(excel, sheet_name='42', index=False)
excel.save()


# graf optimized models
plt.figure(figsize=(15, 15))
plt.gcf().subplots_adjust(left=0.3, bottom=0.3,
                          right=0.7, top=0.7, wspace=0.2, hspace=0.7)
plt.subplot(3, 2, 1)
plt.scatter(X_train02, y_train02)
plt.scatter(X_test02, y_test02)
plt.scatter(X_test02, predictions_forest_op02, c='r', label='RFR')
plt.legend()
plt.xlabel('date')
plt.ylabel('waste generation (in tons)')
plt.title('ASE1_Household waste')
plt.subplot(3, 2, 2)
plt.scatter(X_train12, y_train12)
plt.scatter(X_test12, y_test12)
plt.scatter(X_test02, predictions_forest_op12, c='r', label='RFR')
plt.legend()
plt.xlabel('date')
plt.ylabel('waste generation (in tons)')
plt.title('ASE2_Household waste')
plt.subplot(3, 2, 3)
plt.scatter(X_train22, y_train22)
plt.scatter(X_test22, y_test22)
plt.scatter(X_test22, predictions_svr_op22, c='r', label='SVR')
plt.legend()
plt.xlabel('date')
plt.ylabel('waste generation (in tons)')
plt.title('ASE3_Household waste')
plt.subplot(3, 2, 4)
plt.scatter(X_train32, y_train32)
plt.scatter(X_test32, y_test32)
plt.scatter(X_test32, predictions_poly_op32, c='r', label='Poly')
plt.legend()
plt.xlabel('date')
plt.ylabel('waste generation (in tons)')
plt.title('ASE4_Household waste')
plt.subplot(3, 2, 5)
plt.scatter(X_train42, y_train42)
plt.scatter(X_test42, y_test42)
plt.scatter(X_test42, predictions_svr_op42, c='r', label='SVR')
plt.legend()
plt.xlabel('date')
plt.ylabel('waste generation (in tons)')
plt.title('ASE5_Household waste')


