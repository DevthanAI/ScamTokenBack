import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Read the data
data = pd.read_csv('data_csv/crypto-data.csv')

# remove \n from Name column
data['Name'] = data['Name'].str.replace('\n', '')

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

# use the elbow method to find the optimal number of clusters
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(data)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.savefig('k_calc_img/elbow-method.png')
plt.show()
plt.close()

# use the silhouette method to find the optimal number of clusters
from sklearn.metrics import silhouette_score
silhouette_scores = []
for n_cluster in range(2, 11):
    silhouette_scores.append(silhouette_score(data, KMeans(n_clusters=n_cluster).fit_predict(data)))
plt.plot(range(2, 11), silhouette_scores)
plt.title('The Silhouette Method')
plt.xlabel('Number of clusters')
plt.ylabel('Silhouette Score')
plt.savefig('k_calc_img/silhouette-method.png')
plt.show()
plt.close()

# cluster the data with dbscan
kmeans = KMeans(n_clusters=4, init='k-means++', random_state=42)
kmeans.fit(data)

# create a new column for the cluster
data['Cluster'] = kmeans.labels_ 

# save the data
data.to_csv('data_csv/crypto-data-clustered.csv', index=False)

# show the data with the clusters with x = Market Cap and y = Price
plt.figure(figsize=(10, 6))
plt.scatter(data['Market Cap'], data['Price'], c=data['Cluster'], cmap='rainbow')
plt.xlabel('Market Cap')
plt.ylabel('Price')

# save the plot as jpg with x = Market Cap and y = Price
plt.savefig('clustered_img/market-cap-price.png')
plt.show()
plt.close()

# show the data with the clusters with x = Volume(24h) and y = Price
plt.figure(figsize=(10, 6))
plt.scatter(data['Volume(24h)'], data['Price'], c=data['Cluster'], cmap='rainbow')
plt.xlabel('Volume(24h)')
plt.ylabel('Price')

# save the plot as jpg with x = Volume(24h) and y = Price
plt.savefig('clustered_img/volume-price.png')
plt.show()
plt.close()

# show the data with the clusters with x = % 1h and y = Price
plt.figure(figsize=(10, 6))
plt.scatter(data['% 1h'], data['Price'], c=data['Cluster'], cmap='rainbow')
plt.xlabel('% 1h')
plt.ylabel('Price')

# save the plot as jpg with x = % 1h and y = Price
plt.savefig('clustered_img/percent-1h-price.png')
plt.show()
plt.close()

# show the data with the clusters with x = % 24h and y = Price
plt.figure(figsize=(10, 6))
plt.scatter(data['% 24h'], data['Price'], c=data['Cluster'], cmap='rainbow')
plt.xlabel('% 24h')
plt.ylabel('Price')

# save the plot as jpg with x = % 24h and y = Price
plt.savefig('clustered_img/percent-24h-price.png')
plt.show()
plt.close()

# show the data with the clusters with x = % 7d and y = Price
plt.figure(figsize=(10, 6))
plt.scatter(data['% 7d'], data['Price'], c=data['Cluster'], cmap='rainbow')
plt.xlabel('% 7d')
plt.ylabel('Price')

# save the plot as jpg with x = % 7d and y = Price
plt.savefig('clustered_img/percent-7d-price.png')
plt.show()
plt.close()
