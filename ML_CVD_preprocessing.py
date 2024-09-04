import pandas as pd
from sklearn.preprocessing import PowerTransformer, OrdinalEncoder, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from pyod.models.ecod import ECOD

# Load data
df = pd.read_csv("./CVD_Code/EHS_data2.csv") ##load dataset in
print(df.head()) ## view head for first idea

# Remove NAs in relevant variables
df = df.dropna(subset = ['Sex', 'Age35g', 'BMI', 'whval', 'cholval3', 'hdlval3', 'glyhbval2', 'iffcval2', 'omdiaval', 'omsysval', 'ommapval', 'ompulval', 'dnoft3', 'cvd3a', 'padsympt', 'cigdyal', 'cotval', 'TotmWalWk', 'TotmSitWk', 'TotmModWk', 'TotmVigWk'])
df = df.reset_index(drop=True)

# Pre-processing: Setting up transformers
categorical_transformer_onehot = Pipeline(
    steps=[
        ("encoder", OneHotEncoder(handle_unknown="ignore", drop="first", sparse=False))
    ])

categorical_transformer_ordinal = Pipeline(
    steps=[
        ("encoder", OrdinalEncoder())
    ])

num = Pipeline(
    steps=[
        ("encoder", PowerTransformer())
    ])

# Using column transformer on dataset 
preprocessor  = ColumnTransformer(transformers = [                      #set up how each column should be transformed
                ('cat_onehot', categorical_transformer_onehot, ["Sex"]),
                ('cat_ordinal', categorical_transformer_ordinal, ["Age35g", "cvd3a", "padsympt", "dnoft3"]),
                ('num', num, ["BMI", "whval", "cholval3", "hdlval3", "glyhbval2", "iffcval2", "omdiaval", "omsysval", "ommapval", "ompulval", "cigdyal", "cotval", "TotmWalWk", "TotmSitWk", "TotmModWk", "TotmVigWk"])
                ])

pipeline = Pipeline( 
    steps=[("preprocessor", preprocessor)]
    )
pipe_fit = pipeline.fit(df)

# Fit data with transformer
column_names = ['oneh_Sex', 'cat_Age35g', 'cat_cvd3a', 'cat_padsympt', 'cat_dnoft3', 'num_BMI', 'num_whval', 'num_cholval3', 'num_hdlval3', 'num_glyhbval2', 'num_iffcval2', 'num_omdiaval', 'num_omsysval', 'num_ommapval', 'num_ompulval', 'num_cigdyal', 'num_cotval', 'num_TotmWalWk', 'num_TotmSitWk', 'num_TotmModWk', 'num_TotmVigWk']
data = pd.DataFrame(pipe_fit.transform(df), columns = column_names) #need to figure out a better method to get feature names out

# Stick transformed columns into original data frame
df1 = df.merge(data, right_index=True, left_index=True)

# Determine outliers and remove
clf = ECOD()
clf.fit(data)
outliers = clf.predict(data)
df1["outliers"] = outliers

# Data without outliers
data_no_outliers = df1[df1["outliers"] == 0]
data_no_outliers = data_no_outliers.drop(["outliers"], axis = 1)

# Data with Outliers
data_with_outliers = df1.copy()
data_with_outliers = data_with_outliers.drop(["outliers"], axis = 1)

#print(data_no_outliers.shape)
#print(data_with_outliers.shape)

# Write out data with outliers and without
data_no_outliers.to_csv('dftnooutliers.csv', index=False)
data_with_outliers.to_csv('dftwithoutliers.csv', index=False)

