import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import hdbscan
from sklearn.decomposition import PCA
import pickle

# Read the data
data = pd.read_csv('csv/top-crypto.csv')

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

# create a clusterer using HDBSCAN
clusterer = hdbscan.HDBSCAN(min_cluster_size=5, gen_min_span_tree=True)
clusterer.fit(data)

# add the cluster labels to the dataframe
data['Cluster'] = clusterer.labels_

# calculate the percentage of tokens in each cluster
print(data['Cluster'].value_counts(normalize=True))

# plot the clusters
plt.scatter(data['Market Cap'], data['Price'], c=data['Cluster'], cmap='rainbow')
plt.xlabel('Market Cap')
plt.ylabel('Price')
plt.savefig('img/hdbscan-pca.png')
plt.show()
plt.close()

# save the dataframe to a csv file
data.to_csv('csv/hdbscan-pca.csv', index=False)

# save the model to a pickle file
pickle.dump(clusterer, open('pickle/hdbscan-pca.pkl', 'wb'))