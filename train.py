import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, explained_variance_score, median_absolute_error, r2_score
from sklearn.linear_model import BayesianRidge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
#from sklearn.model_selection import GridSearchCV

def load_data(path):
    fn=pd.read_csv(path)
    return fn

data_file=load_data('Datasets/flights.csv').sample(100000, replace=False)
data_file.reset_index(drop=True, inplace=True)

def preprocessing(file):
    #junk features
    file.drop(['CANCELLATION_REASON','AIR_SYSTEM_DELAY','SECURITY_DELAY','WEATHER_DELAY'], axis=1, inplace=True)
    #feature selection
    cols_to_keep=['DAY','AIRLINE','FLIGHT_NUMBER','DESTINATION_AIRPORT','ORIGIN_AIRPORT','DAY_OF_WEEK','TAXI_OUT','DEPARTURE_DELAY','ARRIVAL_DELAY']
    file=file[cols_to_keep]
    #null value removal
    file.dropna(axis=0, how="any", inplace=True)
    #converting mixed (object) datatypes to string
    for x in file.columns:
        if str(file[x].dtype)=='object':
            file[x]=file[x].astype('string')
    return file

data_file=preprocessing(data_file)
data_file.reset_index(drop=True, inplace=True)

oe=OrdinalEncoder()
def encoding(file):
    df_obj=file.select_dtypes(include=['string'])
    fit=oe.fit_transform(df_obj)
    df_obj=pd.DataFrame(fit, columns=df_obj.columns)
    for x in df_obj.columns:
        file[x]=df_obj[x]
    return file

df_enc=encoding(data_file)

X = df_enc.iloc[:, :-2]
y1 = df_enc["DEPARTURE_DELAY"]
y2 = df_enc["ARRIVAL_DELAY"]

#Scaling features
scale=StandardScaler()
X_scaled=pd.DataFrame(scale.fit_transform(X), columns=X.columns)

def splitting_dataset(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=10)
    return X_train, X_test, y_train, y_test

######### 1st Section: Target Variable: DEPARTURE_DELAY (y1)
X_train, X_test, y_train, y_test = splitting_dataset(X_scaled, y1)
X_train.isnull().sum()
metric1={}

### Decision Tree Regression
dct1=DecisionTreeRegressor(max_depth=12)
dct1.fit(X_train, y_train)
y_pred=dct1.predict(X_test)

a1=mean_squared_error(y_test, y_pred)
b1=mean_absolute_error(y_test, y_pred)
c1=explained_variance_score(y_test, y_pred)
d1=median_absolute_error(y_test, y_pred)
e1=r2_score(y_test, y_pred)

metric1['Decision Tree']=[a1,b1,c1,d1,e1]

### Bayesian Ridge Regression
br1=BayesianRidge(normalize=False, verbose=5)
br1.fit(X_train, y_train)
y_pred = br1.predict(X_test)

a2=mean_squared_error(y_test, y_pred)
b2=mean_absolute_error(y_test, y_pred)
c2=explained_variance_score(y_test, y_pred)
d2=median_absolute_error(y_test, y_pred)
e2=r2_score(y_test, y_pred)

metric1['Bayesian Ridge']=[a2,b2,c2,d2,e2]

### Random Forest Regressor

# #Hyperparameter tuning
# param_grid = {
#     'bootstrap': [False],
#     'max_depth': [13,14,15,16,17],
#     'max_features': ["sqrt"],
#     'min_samples_leaf': [5,6,7],
#     'min_samples_split': [2, 3],
#     'n_estimators': [125, 150, 175, 200]
# }
# # Create a based model
# r = RandomForestRegressor()
# # Instantiate the grid search model
# grid_search = GridSearchCV(estimator = r, param_grid = param_grid, cv = 3, n_jobs = -1, verbose = 5)
#
# grid_search.fit(X_train, y_train)
# grid_search.best_params_

rf1=RandomForestRegressor(n_estimators=150, min_samples_split=3, max_features='sqrt', max_depth=17, min_samples_leaf=7, bootstrap=False, verbose=5)
rf1.fit(X_train, y_train)
y_pred = rf1.predict(X_test)

#{'bootstrap': False, 'max_depth': 17, 'max_features': 'sqrt', 'min_samples_leaf': 7, 'min_samples_split': 3, 'n_estimators': 150}
#rf1=RandomForestRegressor(n_estimators=100, min_samples_split=2, max_features='sqrt', max_depth=15, bootstrap=False, verbose=5)
#rf1=RandomForestRegressor(n_estimators=50, min_samples_split=2, max_features='sqrt', max_depth=15, verbose=5)
#rf1=RandomForestRegressor(n_estimators=50, min_samples_split=2, max_depth=15, verbose=5)

a3=mean_squared_error(y_test, y_pred)
b3=mean_absolute_error(y_test, y_pred)
c3=explained_variance_score(y_test, y_pred)
d3=median_absolute_error(y_test, y_pred)
e3=r2_score(y_test, y_pred)

metric1['Random Forest']=[a3,b3,c3,d3,e3]

# import pickle
# pickle.dump(rf1, open('rf1_0.03399', 'wb'))

# with open("rf1.txt","wb") as fp:
#     pickle.dump(metrics1, fp)

### Gradient Boosting Regressor
gb1=GradientBoostingRegressor()
gb1.fit(X_train, y_train)
y_pred = gb1.predict(X_test)

a4=mean_squared_error(y_test, y_pred)
b4=mean_absolute_error(y_test, y_pred)
c4=explained_variance_score(y_test, y_pred)
d4=median_absolute_error(y_test, y_pred)
e4=r2_score(y_test, y_pred)

metric1['Gradient Boosting']=[a4,b4,c4,d4,e4]

######### 2nd Section: Target Variable: ARRIVAL_DELAY (y2)
X_train, X_test, y_train, y_test = splitting_dataset(X_scaled, y2)
X_train.isnull().sum()
metric2={}

### Decision Tree Regression
dct2=DecisionTreeRegressor(max_depth=12)
dct2.fit(X_train, y_train)
y_pred=dct2.predict(X_test)

a1=mean_squared_error(y_test, y_pred)
b1=mean_absolute_error(y_test, y_pred)
c1=explained_variance_score(y_test, y_pred)
d1=median_absolute_error(y_test, y_pred)
e1=r2_score(y_test, y_pred)

metric2['Decision Tree']=[a1,b1,c1,d1,e1]

### Bayesian Ridge Regression
br2=BayesianRidge(normalize=False, compute_score=True, verbose=5)
br2.fit(X_train, y_train)
y_pred = br2.predict(X_test)

a2=mean_squared_error(y_test, y_pred)
b2=mean_absolute_error(y_test, y_pred)
c2=explained_variance_score(y_test, y_pred)
d2=median_absolute_error(y_test, y_pred)
e2=r2_score(y_test, y_pred)

metric2['Bayesian Ridge']=[a2,b2,c2,d2,e2]

### Random Forest Regressor
# #Hyperparameter tuning
# param_grid = {
#     'bootstrap': [True],
#     'max_depth': [8, 9, 10, 11, 12],
#     'max_features': ["log2"],
#     'min_samples_leaf': [5,6,7],
#     'min_samples_split': [3, 4, 5],
#     'n_estimators': [140, 160, 180]
# }
# # Create a based model
# r = RandomForestRegressor()
# # Instantiate the grid search model
# grid_search = GridSearchCV(estimator = r, param_grid = param_grid, cv = 3, n_jobs = -1, verbose = 5)
#
# grid_search.fit(X_train, y_train)
# grid_search.best_params_

rf2=RandomForestRegressor(bootstrap= True, max_depth= 12, max_features= 'log2', min_samples_leaf=7, min_samples_split= 3, n_estimators= 180, verbose=5)
rf2.fit(X_train, y_train)
y_pred = rf2.predict(X_test)

a3=mean_squared_error(y_test, y_pred)
b3=mean_absolute_error(y_test, y_pred)
c3=explained_variance_score(y_test, y_pred)
d3=median_absolute_error(y_test, y_pred)
e3=r2_score(y_test, y_pred)

metric2['Random Forest']=[a3,b3,c3,d3,e3]
#rf2=RandomForestRegressor(bootstrap= True, max_depth= 10, max_features= 'log2', min_samples_split= 2, n_estimators= 100, verbose=5)

### Gradient Boosting Regressor
gb2=GradientBoostingRegressor(n_estimators=50, verbose=5)
gb2.fit(X_train, y_train)
y_pred = gb2.predict(X_test)

a4=mean_squared_error(y_test, y_pred)
b4=mean_absolute_error(y_test, y_pred)
c4=explained_variance_score(y_test, y_pred)
d4=median_absolute_error(y_test, y_pred)
e4=r2_score(y_test, y_pred)

metric2['Gradient Boosting']=[a4,b4,c4,d4,e4]

# import pickle
# pickle.dump(metric1,open('metric1_100k','wb'))
# pickle.dump(metric2,open('metric2_100k','wb'))
