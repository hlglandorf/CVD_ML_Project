import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics import calinski_harabasz_score
from sklearn.metrics import davies_bouldin_score
from yellowbrick.cluster import KElbowVisualizer

# Load in dataframe from pca with 7 components
df = pd.read_csv("df_pca.csv") 
print(df.T)

# Select columns/features of interest
df1 = df[['Comp1', 'Comp2', 'Comp3', 'Comp4', 'Comp5', 'Comp6', 'Comp7']]
print(df1.T)

# K-means clustering 
# Find best number of clusters first
kmeans = KMeans(n_init=10)
visualizer = KElbowVisualizer(kmeans, k=(2,13))

visualizer.fit(df1)
visualizer.show()

noclusters = 7 #based on the elbow visualiser, the best number of clusters is 6

# Carry out algorithm with best number of clusters
kmeans_set = KMeans(n_clusters=noclusters, n_init=10, random_state=33)
y = kmeans_set.fit_predict(df1)

df['Cluster'] = y

print(df.head())

# Explore clusters 
print(df.groupby(['Cluster']).mean().T)

# Evaluate clusters
print(f"Davies bouldin score: {davies_bouldin_score(df1,y)}")
print(f"Calinski Score: {calinski_harabasz_score(df1,y)}")
print(f"Silhouette Score: {silhouette_score(df1,y)}")

# The results look slightly better than those from the k-means clustering solution without prior pca

