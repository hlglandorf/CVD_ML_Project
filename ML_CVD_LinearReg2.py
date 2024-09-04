from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from mlxtend.feature_selection import ExhaustiveFeatureSelector as EFS
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load in cleaned dataframe from DBSCAN clustering
dfa = pd.read_csv("df_dbscanclusters.csv")
dfa = dfa[dfa.DB_Cluster != -1] #remove noise points (NAs have already been removed during DBSCAN)

# Linear Regression
# Correlation table
print(dfa[['BMI', 'whval', 'cholval3', 'hdlval3', 'glyhbval2', 'iffcval2', 'omdiaval', 'omsysval', 'ommapval', 'ompulval', 'cigdyal', 'cotval', 'TotmWalWk', 'TotmSitWk', 'TotmModWk', 'TotmVigWk']].corr())
# Only used as overview here since the SFS will help with picking features 

# Sequential Feature Selection (SFS)
# Define X and y for SFS
X = dfa[['num_BMI', 'num_whval', 'num_cholval3', 'num_hdlval3', 'num_omdiaval', 'num_omsysval', 'num_ommapval', 'num_ompulval', 'num_cigdyal', 'num_cotval', 'num_TotmWalWk', 'num_TotmSitWk', 'num_TotmModWk', 'num_TotmVigWk']] #choose predictors
y = dfa['iffcval2'] #choose outcome
# Define SFS
effs = EFS(LinearRegression(),
         min_features=2,
         max_features=10,
         scoring='r2',
         cv=0)
effs.fit(X, y)
print(list(effs.best_feature_names_))
columns = list(effs.best_feature_names_)

# Plot the data
#sns.pairplot(dfa)
#plt.show()

# Set up linear regression
model = LinearRegression() #define model

# Define X and y
X = dfa[columns] #choose predictors based on SFS run above
y = dfa['iffcval2'] #choose outcome

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, train_size=0.3)

# Fit the model
model.fit(X_train, y_train)

# View coefficients and intercepts
print(model.coef_)
print(model.intercept_)

# Check how well the model fits the training data
train_predictions = model.predict(X_train)
r2_train = r2_score(y_train, train_predictions)
r2adj_train = 1 - (1-r2_train)*(len(y_train)-1)/(len(y_train)-X_train.shape[1]-1)
rmse_train = mean_squared_error(y_train, train_predictions, squared=False)
print('the r2 for the training set is: ', r2_train)
print('the adjusted r2 for the training set is: ', r2adj_train)
print('the rmse for the training set is: ', rmse_train)

# Check how well the model fits the testing data
predictions = model.predict(X_test)
r2_test = r2_score(y_test, predictions)
r2adj_test = 1 - (1-r2_test)*(len(y_test)-1)/(len(y_test)-X_test.shape[1]-1)
rmse = mean_squared_error(y_test, predictions, squared=False)

print('the r2 for the testing set is: ', r2_test)
print('the adjusted r2 for the testing set is: ', r2adj_test)
print('the rmse for the testing set is: ', rmse)

# R-squared is a lot better after carrying out a dbcan and removing noise points 
# SFS ran but did not improve the r2 in the testing set all that much