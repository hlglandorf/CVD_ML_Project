import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from kneed import KneeLocator
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

# Load in cleaned dataframe from preprocessing
df = pd.read_csv("dftwithoutliers.csv") #outliers have not been removed from this data yet
print(df.T)

# Select variables (these are based on the regression model)
df1 = df[['num_cholval3', 'num_whval', 'num_BMI', 'num_omsysval', 'num_ompulval', 'cat_dnoft3', 'num_iffcval2']] #correlational plots are within the linear regression code

# Implement DBSCAN clustering: intention is to find some more outliers here
# Find the best epsilon first via nearest neighbor algorithm
nearest_neighbors = NearestNeighbors(n_neighbors=7)
neighbors = nearest_neighbors.fit(df1)

distances, indices = neighbors.kneighbors(df1)
distances = np.sort(distances[:,6], axis=0)

# Determine best epsilon 
i = np.arange(len(distances))
knee = KneeLocator(i, distances, S=1, curve='convex', direction='increasing', interp_method='polynomial')

fig = plt.figure(figsize=(5, 5))
knee.plot_knee()
plt.xlabel("Points")
plt.ylabel("Distance")

print(distances[knee.knee])
best_eps = distances[knee.knee]

# Now, use the epsilon for the algorithm
dbscan_cluster1 = DBSCAN(eps=best_eps, min_samples=6)
dbscan_cluster1.fit(df1)

# Number of Clusters
labels=dbscan_cluster1.labels_
N_clus=len(set(labels))-(1 if -1 in labels else 0)
print('Estimated no. of clusters: %d' % N_clus)

# Identify Noise
n_noise = list(dbscan_cluster1.labels_).count(-1)
print('Estimated no. of noise points: %d' % n_noise)

# Adding the labels to the dataframe
df['DB_Cluster'] = labels
print(df.T)

# Explore clusters 
print(df.groupby(['DB_Cluster']).mean().T) 
# -1 is the identifier for noise, so can be removed as outliers
# Then there are two clusters (0 and 1) that seem to differ mostly in the biomarkers 
# The second cluster seems to do generally worse in terms of biomarkers and has a slightly higher HbA1c levels but still normal

# Write out dataframe to use in subsequent analyses
df.to_csv('df_dbscanclusters.csv', index=False)
