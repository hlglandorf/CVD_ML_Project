# Machine Learning Project for Identification of Individuals at Risk for Cardiovascular Disease
This project explores data on cardiovascular disease and fits a number of machine learning algorithms to explore whether 'at risk' cases can be identified through examinations of lifestyle factors and biomarker values. 

The data for this project is from UK Data Services and their dataset 'Health Survey for England' from 2017. The data can be requested from UK Data Services on their website: https://doi.org/10.5255/UKDA-Series-2000021. Note that only a selection of data is used in the project here as the project focuses mainly on biomarkers - this selection is made in the first code file (CVD Exploration). 

The code files in this project are separated into the following files that are explained in further detail below:

CVD Exploration
1. Explores that data to get a general idea of what the data looks like and which variables might be of interest
2. Removes some none responses that are coded numerically (e.g., where 'I don't know' is coded as 97)
3. Pre-selection of columns of interest
4. Writes out a dataframe that is clean with respect to column values that should not appear in the dataset for further analyses. Missing values and outliers are not yet removed.

CVD Pre-processing
1. Removes missing values in relevant columns
2. Transforms data (onehotencoding for binary variables, ordinal transformation for ordinal data, normalisation for numerical data) and adds transformation to dataframe
3. Identifies outliers based on Empirical Cumulative Distribution Function
4. Writes out two dataframes: (1) has outliers removed based on outlier identification, (2) has outliers still included (to allow for other outlier identification methods)

Linear Regression (1)
1. Fits linear regression model to predict HbA1c (three-month glucose concentration in blood)
2. Runs based on file (1) from pre-processing (outliers removed based on ECD function)
3. Selects features of interest based on correlation table and to what extent they correlate with HbA1c value
4. Evaluates fit of model

DBSCAN
1. Carries out DBSCAN based on file (2) from pre-processing (outliers not removed)
2. Uses nearest neighbour algorithm to optimise clustering
3. Identifies outliers for removal in further analyses (see Linear Regression 2 below) and writes out dataframe with identified noise points and clusters
4. Explores identified clusters

Linear Regression (2)
1. Fits linear regression model to predict HbA1c (three-month glucose concentration in blood)
2. Runs based on dataframe with noise points identified by DBSCAN (see above)
3. Uses sequential feature selection for feature selection
4. Evaluates fit of model (and compares this to the model in Linear Regression 1 above)
