import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import PIL

# Read the data
data = pd.read_csv('crypto-data.csv')

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

# standardize the data using z-score
data['Market Cap'] = (data['Market Cap'] - data['Market Cap'].mean()) / data['Market Cap'].std()
data['Volume(24h)'] = (data['Volume(24h)'] - data['Volume(24h)'].mean()) / data['Volume(24h)'].std()
data['Price'] = (data['Price'] - data['Price'].mean()) / data['Price'].std()
data['% 1h'] = (data['% 1h'] - data['% 1h'].mean()) / data['% 1h'].std()
data['% 24h'] = (data['% 24h'] - data['% 24h'].mean()) / data['% 24h'].std()
data['% 7d'] = (data['% 7d'] - data['% 7d'].mean()) / data['% 7d'].std()

# remove outliers
data = data[(np.abs(data['Market Cap']) < 3) & (np.abs(data['Volume(24h)']) < 3) & (np.abs(data['Price']) < 3) & (np.abs(data['% 1h']) < 3) & (np.abs(data['% 24h']) < 3) & (np.abs(data['% 7d']) < 3)]

# cluster the data using k-means clustering
kmeans = KMeans(n_clusters=2)
kmeans.fit(data.drop(['Rank', 'Name', 'Platform', 'Circulating Supply', '% 1h', '% 24h', '% 7d'], axis=1))

# create a new column for the cluster
data['Cluster'] = kmeans.labels_ 

# save the data
data.to_csv('crypto-data-clustered.csv', index=False)

# show the data with the clusters 
plt.figure(figsize=(10, 6))
plt.scatter(data['Market Cap'], data['Price'], c=data['Cluster'], cmap='rainbow')
plt.xlabel('Market Cap')
plt.ylabel('Price')

# save the plot as jpg
plt.savefig('crypto-data-clustered.png')

plt.show()
plt.close()