import matplotlib.pyplot as plt
import pandas as pd

# Load data
df = pd.read_csv("./CVD_Code/EHS_data2.csv") 
print(df.head()) ## view head for first idea

# View descriptive statistics
desc1 = df[["EstHt2", "Estwt2", "Wstval", "hipval", "whval", "BMI", "cholval3", "hdlval3", "glyhbval2", "iffcval2", "omdiaval", "omsysval", "ommapval", "ompulval", "Pain17", "Anxiet17", "SCSatis", "GenHelf", "UsualP", "cigdyal", "cotval", "TotmWalWk", "TotmModWk", "TotmVigWk", "TotmSitWk"]].describe()
print(desc1.T) #descriptive statistics for continuous variables 

# Grouped by
print(df.groupby(['Sex']).describe().T) #biological sex
print(df.groupby(['Age35g']).describe().T) #age groups
print(df.groupby(['cvdiahydd2']).describe().T) #CVD presence, doctor diagnosed 

# Histograms for categorical scales
# Biological sex
plt.hist(df[["Sex"]])
#plt.show()

# Age groups
plt.hist(df[["Age35g"]])
#plt.show()

# Highest educational qualification
plt.hist(df[["topqual3"]])
#plt.show()

# Total household income grouped
plt.hist(df[["HHInc2"]])
#plt.show() ##will need to remove the 96 and 97s from the dataset (represent don't knows and refused)

# Alcohol consuption frequency
plt.hist(df[["dnoft3"]])
#plt.show()

# Fruit and vegetable consumption
# Highest educational qualification
plt.hist(df[["PorFV05b"]])
#plt.show()

# Acute sickness from the last two weeks
plt.hist(df[["acutill"]])
#plt.show()

# Presence of a limiting longlasting illness
plt.hist(df[["limlast"]])
#plt.show()

# Day to day  activities reduced due to illness
plt.hist(df[["ReducAct"]])
#plt.show()

# How long day-to-day activities have been reduced
plt.hist(df[["AffLng"]])
#plt.show()

# Whether taking medication
plt.hist(df[["medcnj"]])
#plt.show()

# Whether have any pain or discomfort 
plt.hist(df[["AnyPain"]])
#plt.show()

# Presence of a limiting longlasting illness
plt.hist(df[["More3m"]])
#plt.show()

# Pain intensity
plt.hist(df[["Painint"]])
#plt.show() ##can be removed, just served as a pre-question to other pain questions

# CVD measure of severity
plt.hist(df[["cvd3a"]])
#plt.show() ##probably a good one to use as an outcome (alternatively CVD3, CVD7, CVD8, CVD4)

# Smoking status
plt.hist(df[["cigst1"]])
#plt.show()

# Symtoms suggestive of Peripheral Arterial Disease (PAD)
plt.hist(df[["padsympt"]])
#plt.show()
print(df[["padsympt"]].describe())


# Removing rows of 96 and 97 (these are where answers were refused)
df1 = df[df.HHInc2 != 97]
df1 = df1[df1.HHInc2 != 96]
print(df1.head())

desc2 = df1.describe()
print(desc2.T) 


# Removing duplicates if they exist
print(df1.duplicated(subset=['SerialA']))
print(df1.drop_duplicates(subset=['SerialA']))
desc3 = df1.describe()
print(desc3.T)

# Dropping columns 
df2 = df1.drop(['HHInc2', 'Unnamed: 0'], axis=1) #can add to this if required to drop more
df3 = df2[['SerialA', 'cvd3a', 'padsympt', 'EstHt2', 'Estwt2', 'Wstval', 'hipval', 'whval', 'BMI', 'cholval3', 'hdlval3', 'glyhbval2', 'iffcval2', 'omdiaval', 'omsysval', 'ommapval', 'ompulval', 'dnoft3', 'Pain17', 'Anxiet17', 'SCSatis', 'GenHelf', 'UsualP', 'cigdyal', 'cotval', 'TotmModWk', 'TotmVigWk']]

df3.to_csv('dfnodub.csv', index=False) #this is clean with respect to columns and values that should not be in here, there are still missing values, outliers, and not normalised, which will be taken care of in subsequent code



