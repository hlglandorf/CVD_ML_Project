import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_validate
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE

# Load in cleaned dataframe from DBSCAN clustering and ECOD cleaned
dfa1 = pd.read_csv("df_dbscanclusters.csv")
dfa1 = dfa1[dfa1.DB_Cluster != -1] ##remove noise points (NAs have already been removed during DBSCAN)
dfa2 = pd.read_csv("dftnooutliers.csv") ##ECOD cleaned

# Recode the cvdiahydd2 to be binary 
dfa1['cvdiahydd2'] = dfa1['cvdiahydd2'].map({1:1, 2:0, 3:0})
dfa2['cvdiahydd2'] = dfa2['cvdiahydd2'].map({1:1, 2:0, 3:0})
#print(dfa2.groupby('cvdiahydd2').count())

# Remove missing values in the outcome
dfa1 = dfa1.dropna(subset=['cvdiahydd2'])
dfa2 = dfa2.dropna(subset=['cvdiahydd2'])
#print(dfa1['cvdiahydd2'].isna().sum())

# Define X and y 
X = dfa1[['BMI', 'whval', 'cholval3', 'hdlval3', 'omdiaval', 'omsysval', 'ommapval', 'ompulval', 'cigdyal', 'cotval', 'TotmWalWk', 'TotmSitWk', 'TotmModWk', 'TotmVigWk']] #choose predictors
y = dfa1['cvdiahydd2'] 

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Resample to balance out the training dataset
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# Scale the data
scaler = StandardScaler()
scaler.fit(X_resampled)
X_train = scaler.transform(X_resampled)
X_test = scaler.transform(X_test)

# Apply the algorithm and evaluate
classifier = SVC(kernel = 'rbf', random_state = 42) #rbf works best 
svc_clf = classifier.fit(X_train, y_resampled)

y_pred = classifier.predict(X_test) #check predictions
print("Classification Report: ", metrics.classification_report(y_test, y_pred)) 

# Check cross-validation results
cv_results = cross_validate(svc_clf, X, y, cv=5)
print(cv_results) 

# Examine confusion matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=classifier.classes_)
disp.plot()
plt.show() #doesn't do as well on accuracy as the random forest, but better at recognising risk cases



