import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# open the pickled model
kmeans = pickle.load(open('pickle/kmeans.pkl', 'rb'))

# open the dataset
data = pd.read_csv('csv/sus-label.csv')

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
data = data[data['Price'] != '0.0...04719']
data = data[data['Price'] != '0.0...07364']

# convert data to numeric
data = data.apply(pd.to_numeric, errors='ignore')

# standardize the data using standard scaler
scaler = StandardScaler()
scaler.fit(data.drop(['Rank', 'Name', 'Platform', 'Circulating Supply', 'reputation'], axis=1))
scaled_features = scaler.transform(data. drop(['Rank', 'Name', 'Platform', 'Circulating Supply', 'reputation'], axis=1))

# create a new dataframe with the scaled features
data = pd.DataFrame(scaled_features, columns=['Market Cap', 'Volume(24h)', 'Price', '% 1h', '% 24h', '% 7d'])

# apply PCA to reduce the dimensions of the data
pca = PCA(n_components=2)
pca.fit(data)
x_pca = pca.transform(data)

# create a new dataframe with the PCA features
data = pd.DataFrame(x_pca, columns=['Market Cap', 'Price'])

# predict the clusters
y_kmeans = kmeans.predict(x_pca)

# add the clusters to the dataset
data['Cluster'] = y_kmeans

# calculate the percentage of tokens in each cluster
print(data['Cluster'].value_counts(normalize=True))

# save the dataset
data.to_csv('csv/sus-kmeans.csv', index=False)

