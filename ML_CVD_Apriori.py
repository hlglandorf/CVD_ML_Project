import pandas as pd 
from mlxtend.frequent_patterns import apriori, association_rules 

# Load cleaned dataframe without outliers
dfa = pd.read_csv("dftnooutliers.csv")

# Select variables of interest
dfa1 = dfa[['BMI', 'cholval3', 'hdlval3', 'glyhbval2', 'omdiaval', 'omsysval', 'medcnj', 'AnyPain', 'cvdiahydd2', 'padsympt', 'angidef', 'bp1', 'cigst1']]
print(dfa1.head())

# Recoding continuous variables to match the binary requirement for apriori 
dfa1.loc[dfa1['BMI'] < 25, 'BMI'] = 0
dfa1.loc[dfa1['BMI'] >= 25, 'BMI'] = 1
dfa1.loc[dfa1['cholval3'] < 5, 'cholval3'] = 0
dfa1.loc[dfa1['cholval3'] >= 5, 'cholval3'] = 1
dfa1.loc[dfa1['hdlval3'] <= 1.1, 'hdlval3'] = 1
dfa1.loc[dfa1['hdlval3'] > 1.1, 'hdlval3'] = 0
dfa1.loc[dfa1['glyhbval2'] < 5.7, 'glyhbval2'] = 0
dfa1.loc[dfa1['glyhbval2'] >= 5.7, 'glyhbval2'] = 1
dfa1.loc[dfa1['omdiaval'] < 80, 'omdiaval'] = 0
dfa1.loc[dfa1['omdiaval'] >= 80, 'omdiaval'] = 1
dfa1.loc[dfa1['omsysval'] < 120, 'omsysval'] = 0
dfa1.loc[dfa1['omsysval'] >= 120, 'omsysval'] = 1

# Recoding categorical variables to match the binary requirement for apriori
dfa1['medcnj'] = dfa1['medcnj'].map({1:1, 2:0})
dfa1['AnyPain'] = dfa1['AnyPain'].map({1:1, 2:0})
dfa1['cvdiahydd2'] = dfa1['cvdiahydd2'].map({1:1, 2:0, 3:0})
dfa1['padsympt'] = dfa1['padsympt'].map({1:0, 2:1, 3:1})
dfa1['angidef'] = dfa1['angidef'].map({1:1, 2:0})
dfa1['bp1'] = dfa1['bp1'].map({1:1, 2:0})
dfa1['cigst1'] = dfa1['cigst1'].map({1:0, 2:1, 3:1, 4:1})

# Check data 
print(dfa1.head())
dfa1 = dfa1.dropna() #remove NAs
dfa1 = dfa1.drop(['omdiaval', 'omsysval'], axis=1) #reduce blood pressure variables to one (bp1)
dfa1 = dfa1.astype('bool') #change datatypes to booleans 
dfa2 = dfa1[:500] #reduce data volume to speed up algorithm (can remove this line when running on servers)

# Carry out the apriori algorithm
frq_items = apriori(dfa2, min_support = 0.05, use_colnames = True) 
rules = association_rules(frq_items, metric ="lift", min_threshold = 1) 
rules = rules.sort_values(['confidence', 'lift'], ascending =[False, False]) 

# View associations/rules
print(rules.head()) 
