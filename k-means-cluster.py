import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
import pickle

# Read the data
data = pd.read_csv('csv/crypto-data.csv')

# Remove \r\n in Name column
data['Name'] = data['Name'].str.replace('\r\n', '')

# remove $ from Market Cap, Volume and Price columns
data['Market Cap'] = data['Market Cap'].str.replace('$', '')
data['Volume(24h)'] = data['Volume(24h)'].str.replace('$', '')
data['Price'] = data['Price'].str.replace('$', '')

# remove commas from Market Cap, Volume and Price columns
data['Market Cap'] = data['Market Cap'].str.replace(',', '')
data['Volume(24h)'] = data['Volume(24h)'].str.replace(',', '')
data['Price'] = data['Price'].str.replace(',', '')

# remove % from % 1h, % 24h, and % 7d columns
data['% 1h'] = data['% 1h'].str.replace('%', '')
data['% 24h'] = data['% 24h'].str.replace('%', '')
data['% 7d'] = data['% 7d'].str.replace('%', '')

# remove row with ? in Volume(24h) column
data = data[data['Volume(24h)'] != '?']

# remove row with excessively small values in price column
data = data[data['Price'] != '0.0...05073']
data = data[data['Price'] != '0.0...07413']
data = data[data['Price'] != '0.0...04719']
data = data[data['Price'] != '0.0...07364']

# convert data to numeric
data = data.apply(pd.to_numeric, errors='ignore')

# standardize the data using standard scaler
scaler = StandardScaler()
scaler.fit(data.drop(['Rank', 'Name', 'Platform', 'Circulating Supply'], axis=1))
scaled_features = scaler.transform(data. drop(['Rank', 'Name', 'Platform', 'Circulating Supply'], axis=1))

# create a new dataframe with the scaled features   
data = pd.DataFrame(scaled_features, columns=['Market Cap', 'Volume(24h)', 'Price', '% 1h', '% 24h', '% 7d'])

# remove the outliers using standard deviation
data = data[(np.abs(data['Market Cap'] - data['Market Cap'].mean()) <= (3 * data['Market Cap'].std()))]
data = data[(np.abs(data['Volume(24h)'] - data['Volume(24h)'].mean()) <= (3 * data['Volume(24h)'].std()))]
data = data[(np.abs(data['Price'] - data['Price'].mean()) <= (3 * data['Price'].std()))]
data = data[(np.abs(data['% 1h'] - data['% 1h'].mean()) <= (3 * data['% 1h'].std()))]
data = data[(np.abs(data['% 24h'] - data['% 24h'].mean()) <= (3 * data['% 24h'].std()))]
data = data[(np.abs(data['% 7d'] - data['% 7d'].mean()) <= (3 * data['% 7d'].std()))]

# apply PCA to reduce the dimensions of the data
pca = PCA(n_components=2)
pca.fit(data)
x_pca = pca.transform(data)

# create a new dataframe with the PCA data
data = pd.DataFrame(x_pca, columns=['Market Cap', 'Price'])

# use the elbow method to find the optimal number of clusters
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, n_init=10, init='k-means++', random_state=42)
    kmeans.fit(data)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.savefig('img/elbow-method.png')
plt.show()
plt.close()

# use the silhouette method to find the optimal number of clusters
silhouette_scores = []
for n_cluster in range(2, 11):
    silhouette_scores.append(silhouette_score(data, KMeans(n_init = 10, n_clusters=n_cluster).fit_predict(data)))
plt.plot(range(2, 11), silhouette_scores)
plt.title('The Silhouette Method')
plt.xlabel('Number of clusters')
plt.ylabel('Silhouette Score')
plt.savefig('img/silhouette-method.png')
plt.show()
plt.close()

# cluster the data optimally with infinite iterations
kmeans = KMeans(n_clusters=5, n_init=10, init='k-means++', random_state=42, max_iter=20)
kmeans.fit(data)
data['Cluster'] = kmeans.labels_

# calculate the mean of each cluster
print(data.groupby('Cluster').mean())

# calculate the percentage of tokens in each cluster
print(data['Cluster'].value_counts(normalize=True))

# show the data with the clusters with x = Market Cap and y = Price
plt.figure(figsize=(10, 6))
plt.scatter(data['Market Cap'], data['Price'], c=data['Cluster'], cmap='rainbow', label='Clusters', alpha=0.7)

# show the cluster centers
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=30, c='black', label='Centroids')

# show the text for the cluster centers
for i, txt in enumerate(kmeans.cluster_centers_):
    plt.annotate(i, (kmeans.cluster_centers_[i][0], kmeans.cluster_centers_[i][1]), fontsize=20)

# set the details of the plot
plt.xlabel('Market Cap')
plt.ylabel('Price')
plt.title('Crypto Data Clusters')

# save the image
plt.savefig('img/k-means.png')
plt.show()
plt.close()

# sort the data by highest market cap
data = data.sort_values(by=['Market Cap'], ascending=False)

# save the data to a csv file
data.to_csv('csv/k-means.csv', index=False)

# save the model to a pickle file
pickle.dump(kmeans, open('pickle/k-means.pkl', 'wb'))