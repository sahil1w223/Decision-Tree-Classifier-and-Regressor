import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_diabetes

data = load_diabetes()
data

df = pd.DataFrame(data.data,columns=data.feature_names)

df['target'] = data.target

df.head()

x = df.drop('target',axis=1)
y = df['target']

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

from sklearn.tree import DecisionTreeRegressor
model = DecisionTreeRegressor()
model.fit(x_train,y_train)

from sklearn import tree
plt.figure(figsize=(25,20))
tree.plot_tree(model,filled=True)

from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
y_pred = model.predict(x_test)
print(accuracy_score(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))

param = {
    'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
    'splitter': ['best', 'random'],
    'max_depth': [2, 4, 6, 8, 10, 12],
    'min_samples_split': [2, 4, 6, 8, 10, 12],
    'min_samples_leaf': [1, 2, 3, 4, 5, 6],
    'max_features': ['sqrt', 'log2']
}

from sklearn.model_selection import GridSearchCV
grid = GridSearchCV(estimator=model, param_grid=param, cv=5)

import warnings
warnings.filterwarnings('ignore')

grid.best_params_

grid.fit(x_train,y_train)

from sklearn import tree
plt.figure(figsize=(25,20))
tree.plot_tree(grid.best_estimator_,filled=True)

from sklearn.metrics import r2_score, mean_squared_error
y_pred1 = grid.predict(x_test)
print(f"R2 Score: {r2_score(y_test,y_pred1)}")
print(f"Mean Squared Error: {mean_squared_error(y_test,y_pred1)}")

