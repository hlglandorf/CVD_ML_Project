import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics import calinski_harabasz_score
from sklearn.metrics import davies_bouldin_score
from yellowbrick.cluster import KElbowVisualizer

# Load in cleaned dataframe without outliers
df = pd.read_csv("dftnooutliers.csv")
print(df.T)

# Select columns/features of interest
df1 = df[['oneh_Sex', 'cat_Age35g', 'cat_cvd3a', 'cat_padsympt', 'cat_dnoft3', 'num_BMI', 'num_whval', 'num_cholval3', 'num_hdlval3', 'num_iffcval2', 'num_omdiaval', 'num_omsysval', 'num_ommapval', 'num_ompulval', 'num_cigdyal', 'num_cotval', 'num_TotmWalWk', 'num_TotmSitWk', 'num_TotmModWk', 'num_TotmVigWk']]
print(df1.T)

# K-means clustering 
# Find best number of clusters first
kmeans = KMeans(n_init=10)
visualizer = KElbowVisualizer(kmeans, k=(2,10))

visualizer.fit(df1)
visualizer.show()

noclusters = 5

# Carry out algorithm with best number of clusters
kmeans_set = KMeans(n_clusters=noclusters, n_init=10)
y = kmeans_set.fit_predict(df1)

df['Cluster'] = y

print(df.head())

# Explore clusters 
print(df.groupby(['Cluster']).mean().T)

# Evaluate clusters
print(f"Davies bouldin score: {davies_bouldin_score(df1,y)}")
print(f"Calinski Score: {calinski_harabasz_score(df1,y)}")
print(f"Silhouette Score: {silhouette_score(df1,y)}")

# Evaluation metrics are not looking that good, feature selection/reduction prior to algorithm is required

