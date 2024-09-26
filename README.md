# Machine Learning Project for Identification of Individuals at Risk for Cardiovascular Disease
This project explores data on cardiovascular disease and fits a number of machine learning algorithms to explore whether 'at risk' cases can be identified through examinations of lifestyle factors and biomarker values. 

The data for this project is from UK Data Services and their dataset 'Health Survey for England' from 2017. The data can be requested from UK Data Services on their website: https://doi.org/10.5255/UKDA-Series-2000021. Note that only a selection of data is used in the project here as the project focuses mainly on biomarkers - this selection is made in the first code file (CVD Exploration). 

The code files in this project are separated into the following files that are explained in further detail below:

(1) CVD Exploration
1. Explores that data to get a general idea of what the data looks like and which variables might be of interest
2. Removes some none responses that are coded numerically (e.g., where 'I don't know' is coded as 97)
3. Pre-selection of columns of interest
4. Writes out a dataframe that is clean with respect to column values that should not appear in the dataset for further analyses. Missing values and outliers are not yet removed.

(2) CVD Pre-processing
1. Removes missing values in relevant columns
2. Transforms data (onehotencoding for binary variables, ordinal transformation for ordinal data, normalisation for numerical data) and adds transformation to dataframe
3. Identifies outliers based on Empirical Cumulative Distribution Function
4. Writes out two dataframes: (1) has outliers removed based on outlier identification, (2) has outliers still included (to allow for other outlier identification methods)

(3) Linear Regression (1)
1. Fits linear regression model to predict HbA1c (three-month glucose concentration in blood)
2. Runs based on file (1) from pre-processing (outliers removed based on ECD function)
3. Selects features of interest based on correlation table and to what extent they correlate with HbA1c value
4. Evaluates fit of model

(4) DBSCAN
1. Carries out DBSCAN based on file (2) from pre-processing (outliers not removed)
2. Uses nearest neighbour algorithm to optimise clustering
3. Identifies outliers for removal in further analyses (see Linear Regression 2 below) and writes out dataframe with identified noise points and clusters
4. Explores identified clusters

(5) Linear Regression (2)
1. Fits linear regression model to predict HbA1c (three-month glucose concentration in blood)
2. Runs based on dataframe with noise points identified by DBSCAN (see above)
3. Uses sequential feature selection for feature selection
4. Evaluates fit of model (and compares this to the model in Linear Regression 1 above)

(6) K-means Clustering (1)
1. Carries out K-means clustering on file (1) from pre-processing (outliers removed)
2. Uses Elbow Visualiser to find the best number of clusters
3. Uses K-means clustering algorithm to find clusters
4. Explores clusters and evaluates the fit of the clusters

(7) Principle Component Analysis (PCA)
1. Carries out principle comnponent analysis on file (1) from pre-processing (outliers removed) to reduce the number of features
2. Uses explained variance ratio by number of components to find best number of components
3. Performs PCA on the data
4. Adds components to original dataframe to use in subsequent analyses

(8) K-means Clustering (2)
1. Carries out K-means clustering on components from PCA
2. Uses Elbow Visualiser to find the best number of clusters
3. Uses K-means clustering algorithm to find clusters
4. Explores clusters and evaluates the fit of the clusters

(9) Agglomertative Clustering
1. Carries out agglomertative clustering on file (1) from pre-procesing (outliers removed)
2. Uses dendogram to visualise possible clustering solution and, together with algorithm based on Calinski Harabasz Score, determine best number of clusters
3. Examines clustering solution 

(10) Apriori Algorithm
1. Carries out Aprior algorithm on file (1) from pre-processing (outliers removed)
2. Recoded variables to fit binrary requirement prior to carrying out algorithm
3. Examines determined rules and associations between variables/features

(11) Naive Base Classifier
1. Fits Naive Base classifier on file cleaned based on DBSCAN solution (outliers removed, see above)
2. Uses SMOTE to balance the dataset
3. Fits classifier and uses cross validation to evaluate ability to generalise as well as examines confusion matrix of performance

(12) Random Forest Classifier
1. Fits Random Forest classifier on file (1) from pre-processing (outliers removed)
2. Uses SMOTE to balance the dataset
3. Carries out hyperparameter tuning with grid search
4. Fits classifier and uses confusion matric and classification report to evaluate performance
5. Examines feature importance 

(13) Support Vector Classifier
1. Fits Support Vector Classifier (SVC) on file based on DBSCAN solution (outliers removed, see above)
2. Uses SMOTE to balance the dataset
3. Fits classifier and evaluates the solution based on classification report, cross validation, and confusion matrix

(14) Neural Network to identify CVD risk cases in keras with tensorflow
1. Fits neural network on file (1) from pre-processing (outliers removed)
2. Uses SMOTE to balance the dataset
3. Defines a simple and more complex model architecture (to compare)
4. Creates earlystopping rule to avoid overfitting
5. Compiles parameters for models
6. Trains both models
7. Evaluates testing performance of both models with classification report and confusion matrices as well as examines development of loss over training process to check for overfitting

(15) Neural Network to identify CVD risk cases with pytorch
1. Fits neural network on file (1) from pre-processing (outliers removed)
2. Uses SMOTE to balance the dataset
3. Sets up torch tensors for neural network and initiates training and testing set
4. Defines model architecture
5. Defines parameters for training and testing
6. Trains the model
7. Evaluates the model with classification report and confusion matrix
