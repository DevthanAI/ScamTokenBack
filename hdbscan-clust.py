import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import hdbscan

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

# create a clusterer for hdbscan
clusterer = hdbscan.HDBSCAN(min_cluster_size=5).fit(data)
data['Cluster'] = clusterer.labels_

# calculate the percentage of tokens in each cluster
print(data['Cluster'].value_counts(normalize=True))

# plot the clusters
clusterer.condensed_tree_.plot(select_clusters=True, selection_palette=sns.color_palette('deep', 8))
plt.savefig('img/hdbscan.png')
plt.show()
plt.close()

# save the data to a csv file
data.to_csv('csv/hdbscan.csv', index=False)
