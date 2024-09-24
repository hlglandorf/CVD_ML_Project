import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
from sklearn.metrics import calinski_harabasz_score

# Load in cleaned dataframe without outliers
df = pd.read_csv("dftnooutliers.csv")
print(df.T)

# Select columns/features of interest
df1 = df[['oneh_Sex', 'cat_Age35g', 'cat_cvd3a', 'cat_padsympt', 'cat_dnoft3', 'num_BMI', 'num_whval', 'num_cholval3', 'num_hdlval3', 'num_glyhbval2', 'num_iffcval2', 'num_omdiaval', 'num_omsysval', 'num_ommapval', 'num_ompulval', 'num_cigdyal', 'num_cotval', 'num_TotmWalWk', 'num_TotmSitWk', 'num_TotmModWk', 'num_TotmVigWk']]
print(df1.T)

# Visualise with dendrogram 
linkage_method = linkage(df1, method ='ward', metric='euclidean')
Dendrogram = dendrogram(linkage_method)
plt.show() #looks like two clusters may be best based on visual distance

# Find number of clusters
for n_clusters in range(2, 10):
    clustering = AgglomerativeClustering(n_clusters=n_clusters)
    labels = clustering.fit_predict(df1)
    score = calinski_harabasz_score(df1, labels)
    print(f"For n_clusters = {n_clusters}, the Calinski-Harabasz score is: {score}") #two cluster solution maximises the score, so again, appears to be best solution

# Set model
cluster_ea = AgglomerativeClustering(n_clusters=2, linkage='ward', metric='euclidean') # setting for five clusters
cluster_ea.fit(df1)

# Add labels to dataframe
labels = pd.DataFrame(cluster_ea.labels_, columns=['Agg_Cluster'])
print(labels)
df2 = df1.merge(labels, right_index=True, left_index=True)
print(df2.head())

# Grouping to explore groups
print(df2.groupby(['Agg_Cluster']).mean().T) 



