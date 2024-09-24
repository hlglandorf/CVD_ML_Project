from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import cross_validate
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE

# Load cleaned dataframes from DBSCAN clustering
dfa1 = pd.read_csv("df_dbscanclusters.csv")
dfa1 = dfa1[dfa1.DB_Cluster != -1] ##remove noise points (NAs have already been removed during DBSCAN)

# Recode the cvdiahydd2 to be binary 
dfa1['cvdiahydd2'] = dfa1['cvdiahydd2'].map({1:1, 2:0, 3:0})
#print(dfa1.groupby('cvdiahydd2').count()) #to check

dfa1 = dfa1.dropna(subset=['cvdiahydd2'])
#print(dfa1['cvdiahydd2'].isna().sum()) #to check

# Define X and y 
X = dfa1[['num_BMI', 'num_whval', 'num_cholval3', 'num_hdlval3', 'num_omdiaval', 'num_omsysval', 'num_ommapval', 'num_ompulval', 'num_cigdyal', 'num_cotval', 'num_TotmWalWk', 'num_TotmSitWk', 'num_TotmModWk', 'num_TotmVigWk']] #choose predictors
y = dfa1['cvdiahydd2'] #choose outcome

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Use SMOTE to balance out the dataset
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
print(y_resampled.value_counts()) #to check

# Set the model
naive_bayes = GaussianNB() #using Gaussian here, because the outcome is binary and features continuous

# Fit the data to the classifier
nb_clf = naive_bayes.fit(X_resampled, y_resampled)
y_predicted = naive_bayes.predict(X_test)
print("Classification Report: ", metrics.classification_report(y_test, y_predicted)) 

# Use cross-validation to avoid overfitting
cv_results = cross_validate(nb_clf, X, y, cv=5)
print(cv_results) 

# Examine confusion matrix
cm = confusion_matrix(y_test, y_predicted)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=naive_bayes.classes_)
disp.plot()
plt.show() #the classifier does not do particularly well, room for improvement
