import djclick as click
from scipy import stats
from sqlite3 import sq_access
from sqlite3 import sq_get
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeRegressor
from sklearn.inspection import plot_partial_dependence
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

df = sq_access(init_ , "./preparationData.py", t= pd.DataFrame)

y = df['duration'].values
X = df.drop('duration',axis=1).values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=39)

def lineapi():
  y = df['duration'].values
  X = df.drop('duration',axis=1).values
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=39)
  reg_all = LinearRegression()
  reg_all.fit(X_train,y_train)
  y_pred = reg_all.predict(X_test)
  print("R^2: {}".format(reg_all.score(X_test, y_test)))
  rmse = np.sqrt(mean_squared_error(y_test,y_pred))
  print("Root Mean Squared Error: {}".format(rmse))

def cvapi():
  y = df['duration'].values
  X = df.drop('duration',axis=1).values
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=39)
  cv_scores = cross_val_score(reg_all,X,y,cv=5)
  print(cv_scores)
  print("Average 5-Fold CV Score: {}".format(np.mean(cv_scores)))
  cvscores_10 = cross_val_score(reg_all,X,y,cv=10)
  print(np.mean(cvscores_10))
  lasso = Lasso(alpha=0.4,normalize=False)
  lasso.fit(X,y)
  lasso_coef = lasso.coef_
  print(lasso_coef
  
  
def ridgeapi():
  y = df['duration'].values
  X = df.drop('duration',axis=1).values
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=39)
  alpha_space = np.logspace(-4, 0, 50)
  ridge_scores = []
  ridge_scores_std = []
  ridge = Ridge(normalize=True)
  for alpha in alpha_space:
    ridge.alpha = alpha
    ridge_cv_scores = cross_val_score(ridge,X,y,cv=5)
    ridge_scores.append(np.mean(ridge_cv_scores))
    ridge_scores_std.append(np.std(ridge_cv_scores))
 print(ridge_cv_scores)
 ridge = Ridge(alpha=0.5, normalize=True)
 ridge_cv = cross_val_score(ridge, X, y, cv=5)
 print(ridge_cv
	
def score_regression(algo, X_test, y_test):
    predictions = algo.predict(X_test)
    errors = abs(predictions - y_test)
    print('Mean Absolute Error:', round(np.mean(errors), 2), 'jours.')
    y_test = [1 if value == 0 else value for value in y_test ]
    mape = 100 * (errors / y_test)
    accuracy = 100 - np.mean(mape)
    print('Accuracy:', round(accuracy, 2), '%.')
	

def dtapi():
  y = df['duration'].values
  X = df.drop('duration',axis=1).values
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=39)
  treed = DecisionTreeRegressor()
  treed.fit(X_train,y_train)
  score_regression(treed, X_test,y_test)

def mlpapi():
  y = df['duration'].values
  X = df.drop('duration',axis=1).values
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=39)
  mlp = make_pipeline(StandardScaler(),MLPRegressor(hidden_layer_sizes=(40,30), max_iter=600, random_state=0))
  mlp.fit(X_train, y_train)
  score_regression(mlp, X_test,y_test)

def rfapi():
  y = df['duration'].values
  X = df.drop('duration',axis=1).values
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=39)
  rf = RandomForestRegressor(n_estimators = 1000, random_state = 42,criterion='mse')
  rf.fit(X_train, y_train)
  y_predrf = rf.predict(X_test)  
  print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
  print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
  print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
  
def elastapi():
  y = df['duration'].values
  X = df.drop('duration',axis=1).values
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=39)
  l1_space = np.linspace(0, 1, 30)
  param_grid = {'l1_ratio': l1_space}
  elastic_net = ElasticNet()
  gm_cv = GridSearchCV(elastic_net, param_grid, cv=5)
  gm_cv.fit(X_train, y_train)
  y_pred = gm_cv.predict(X_test)
  r2 = gm_cv.score(X_test, y_test)
  mse = mean_squared_error(y_test, y_pred)
  print("Tuned ElasticNet l1 ratio: {}".format(gm_cv.best_params_))
  print("Tuned ElasticNet R squared: {}".format(r2))
  print("Tuned ElasticNet MSE: {}".format(mse))

