from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load in cleaned dataframes without outliers
df = pd.read_csv("dfnooutliers.csv")
print(df.T)

# Pick columns that are continuous and could, in theory, be related to the HbA1c levels chosen to predict here
df1 = df[['whval', 'BMI', 'cholval3', 'hdlval3', 'glyhbval2', 'iffcval2', 'TotmModWk', 'TotmVigWk', 'omdiaval', 'omsysval', 'ommapval', 'ompulval', 'cotval', 'dnoft3', 'cigdyal']]

# Carry out linear regression
# First examine correlation table
print(df1.corr()) #correlations are assessed to pick predictors for the regression. Any variable that has a correlation higher than .1 is chosen to be included here

# Pick columns for regression
df2 = df1[['cholval3', 'whval', 'BMI', 'omsysval', 'ompulval', 'dnoft3', 'iffcval2']].dropna(axis=0) #missing values in the predictors or outcome variable are removed

# Plot the data to get an impression of relationships
sns.pairplot(df1)
plt.show()

# Set up linear regression model
model = LinearRegression() #define model

# Define X and y
X = df2[['cholval3', 'whval', 'BMI', 'omsysval', 'ompulval', 'dnoft3']] #choose predictors
y = df2['iffcval2'] #choose outcome

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, train_size=0.3) 

# Fit the model
model.fit(X_train, y_train)

# Print coefficients and intercepts
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
print('the adjusted r2 for the training set is: ', r2adj_test)
print('the rmse for the testing set is: ', rmse)
