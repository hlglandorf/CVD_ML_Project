import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE

# Load in data without outliers
dfa2 = pd.read_csv("dftnooutliers.csv") 

# Map the outcome variable to two outcomes/turn it binary
dfa2['cvdiahydd2'] = dfa2['cvdiahydd2'].map({1:1, 2:0, 3:0})

# Make sure no null values are in the outcomes variable 
dfa2 = dfa2.dropna(subset=['cvdiahydd2'])

# Define X and y
X = dfa2[['BMI', 'Sex', 'whval', 'cholval3', 'hdlval3', 'omdiaval', 'omsysval', 'ommapval', 'ompulval', 'cigdyal', 'cotval', 'TotmWalWk', 'TotmSitWk', 'TotmModWk', 'TotmVigWk']] #choose predictors
y = dfa2['cvdiahydd2'] #choose outcome
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2) #train/test split

# Resample the data to match the training cases 
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# Hyperparameter-tuning
rf = RandomForestClassifier(random_state=42, n_jobs=-1)

params = {
    'max_depth': [2,3,5,10,15,20],
    'min_samples_leaf': [5,10,20,50,100,200,300],
    'n_estimators': [10,25,30,50,100,200,300]
}

# Set and run the grid search model
grid_search = GridSearchCV(estimator=rf,
                           param_grid=params,
                           cv = 4,
                           n_jobs=-1, verbose=1, scoring="accuracy")

grid_search.fit(X_resampled, y_resampled)
print(grid_search.best_score_)
rf_best = grid_search.best_estimator_
print(rf_best)

# Apply the classifier
rf_clf = RandomForestClassifier(random_state=42, max_depth=15, min_samples_leaf=5, n_estimators=200, oob_score=True, n_jobs=-1)
rf_clf.fit(X_resampled,y_resampled)

# Evaluate the classifier
y_predict = rf_clf.predict(X_test)
print(classification_report(y_test, y_predict))

# Confusion matrix
cm = confusion_matrix(y_test, y_predict)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show() #accuracy is better than in the naive bayes, but the model is not good at identifying risk cases, needs to do better 

# Display feature importance
imp_df = pd.DataFrame({
    "Varname": X_train.columns,
    "Imp": rf_best.feature_importances_
})

print(imp_df.sort_values(by="Imp", ascending=False))
