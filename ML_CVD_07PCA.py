import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Load dataset without outliers
df = pd.read_csv("dftnooutliers.csv")
print(df.T)

# Select columns/features of interest (using scaled data here)
df1 = df[['oneh_Sex', 'cat_Age35g', 'cat_cvd3a', 'cat_padsympt', 'cat_dnoft3', 'num_BMI', 'num_whval', 'num_cholval3', 'num_hdlval3', 'num_iffcval2', 'num_omdiaval', 'num_omsysval', 'num_ommapval', 'num_ompulval', 'num_cigdyal', 'num_cotval', 'num_TotmWalWk', 'num_TotmSitWk', 'num_TotmModWk', 'num_TotmVigWk']]
print(df1.T)

# Figure out best number of PCs
nums = np.arange(21)

var_ratio = []
for num in nums:
  pca = PCA(n_components=num)
  pca.fit(df1)
  var_ratio.append(np.sum(pca.explained_variance_ratio_))

# Plotting for visualisation
plt.figure(figsize=(4,2),dpi=150)
plt.grid()
plt.plot(nums,var_ratio,marker='o')
plt.xlabel('n_components')
plt.ylabel('Explained variance ratio')
plt.title('n_components vs. Explained Variance Ratio')

plt.show()

# Perform PCA
pca = PCA(n_components=7)
pca.fit_transform(df1)
print(pca.components_)
print(sum(pca.explained_variance_ratio_))

# Store components
df2 = pd.DataFrame(pca.components_)
print(df2.T)

# Transform the data and add to dataframe
df1_trans = pca.transform(df1)
df1_trans = pd.DataFrame(df1_trans)
df1_trans.columns = ['Comp1', 'Comp2', 'Comp3', 'Comp4', 'Comp5', 'Comp6', 'Comp7']
df1_trans.set_axis(['Comp1', 'Comp2', 'Comp3', 'Comp4', 'Comp5', 'Comp6', 'Comp7'], axis='columns') 
print(df1_trans)

#df3 = pd.concat([df, df1_trans], axis=1)
#print(df3.T)
#df3.to_csv('df_pca.csv', index=False)
