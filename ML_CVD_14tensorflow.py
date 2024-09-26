import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Load Data
df1 = pd.read_csv("dftnooutliers.csv")

# Recoding outcome variable to be binary
df1['cvdiahydd2'] = df1['cvdiahydd2'].map({1:1, 2:0, 3:0})

# Drop NAs
df1 = df1.dropna(subset=['cvdiahydd2', 'BMI', 'whval', 'cholval3', 'hdlval3', 'omdiaval', 'omsysval', 'ommapval', 'ompulval', 'cigdyal', 'cotval', 'TotmWalWk', 'TotmSitWk', 'TotmModWk', 'TotmVigWk'])
print(len(df1.index))

# Pick X and y
X = df1[['BMI', 'whval', 'cholval3', 'hdlval3', 'omdiaval', 'omsysval', 'ommapval', 'ompulval', 'cigdyal', 'cotval', 'TotmWalWk', 'TotmSitWk', 'TotmModWk', 'TotmVigWk']] # choose predictors (X)
y = df1['cvdiahydd2'] # choose outcome (y)

# Split the dataset into train and test set 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# The groups (label y) are not equal in size, so resampling is required for training set 
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
y_train = y_resampled # after upsampling, y train needs to be updated

# Scale the data to avoid weighting of features pre model
scaler = StandardScaler()
scaler.fit(X_resampled)
X_train = scaler.transform(X_resampled)
X_test = scaler.transform(X_test)

# Data size check
print("X resampled: ", len(X_resampled))
print("Y resampled: ", len(y_resampled))
print("X train: ", len(X_train))
print("Y train: ", len(y_train))
print("X test: ", len(X_test))
print("Y test: ", len(y_test))

# Build a neural net with tensorflow 
# Start with defining the input shape
input_shape = [X_train.shape[1]]
print(input_shape)

# Create a model (starting linear)
model1 = tf.keras.Sequential([
    tf.keras.Input(shape=input_shape),
    tf.keras.layers.Dense(units=1, activation='sigmoid')
])

# Print out the model summary to assess
print(model1.summary())

# Now, build on the basic model by creating a multilayer model with ReLu activation
model2 = tf.keras.Sequential([
    tf.keras.Input(shape=input_shape),
    tf.keras.layers.Dense(units=64, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(0.001)),
    tf.keras.layers.Dense(units=128, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(0.001)),
    tf.keras.layers.Dropout(rate=0.5),
    tf.keras.layers.Dense(units=64, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(0.001)),
    tf.keras.layers.Dense(units=1, activation='sigmoid')
])

print(model2.summary())

# Set up earlystopping and adam optimiser
callback = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
Adam = tf.keras.optimizers.Adam(learning_rate=0.001)

# Compile other parameters for model
model1.compile(optimizer='adam', # using adam optimizer 
               loss = 'mae', # using mean absolute error
               metrics = ['accuracy']) # asking for accuracy for epochs
model2.compile(optimizer=Adam, # using adam optimizer 
               loss = 'binary_crossentropy', # using binary crossentropy for binary outcome
               metrics = ['accuracy']) # asking for accuracy for epochs

# Model 1
losses1 = model1.fit(X_train, y_train, 
                     validation_data=(X_test, y_test), 
                     batch_size=256, 
                     epochs=100)
# model 1 has about .63 accuracy on train data and .55 on test data

# Examine examples of targets and predictions
targets1=y_test[0:3]
predictions1=model1.predict(X_test[0:3, :])
print(targets1)
print(predictions1)

# Evaluate classification report
y_pred = model1.predict(X_test)
b_pred = (y_pred > 0.5).astype(int)
print("Classification Report: ", metrics.classification_report(y_test, b_pred)) 

# Examine confusion matrix
cm = confusion_matrix(y_test, b_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()

# Model 2
losses2 = model2.fit(X_train, y_train, 
                     validation_data=(X_test, y_test), 
                     batch_size=256, 
                     epochs=100,
                     callbacks=[callback])
# model has about .82 accuracy on train data and .73 on test data
# these are typical numbers for this kind of data, higher quality data with imaging and genetic info could improve accuracy further
# generally, the neural nets are doing better at identifying risk cases compared to the more classic classifiers

# Examine examples of targets and predictions
targets2=y_test[0:3]
predictions2=model2.predict(X_test[0:3, :])
print(targets2)
print(predictions2)

# Evaluate classification report
y_pred2 = model2.predict(X_test)
b_pred2 = (y_pred2 > 0.5).astype(int)
print("Classification Report: ", metrics.classification_report(y_test, b_pred2)) 

# Examine confusion matrix
cm2 = confusion_matrix(y_test, b_pred2)
disp2 = ConfusionMatrixDisplay(confusion_matrix=cm2)
disp2.plot()

# Now analyse the loss and figure out if it is overfitting
loss_df1 = pd.DataFrame(losses1.history) #history stores the loss/val loss in each epoch
loss_df1.loc[:,['loss','val_loss']].plot()
plt.show()

loss_df2 = pd.DataFrame(losses2.history) #history stores the loss/val loss in each epoch
loss_df2.loc[:,['loss','val_loss']].plot()
plt.show()
