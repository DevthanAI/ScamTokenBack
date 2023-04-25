import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import hdbscan

# Read the data
data = pd.read_csv('csv/crypto-data-label.csv')

# Remove \r\n in Name column
data['Name'] = data['Name'].str.replace('\r\n', '')

# remove rows that don't have value for reputation 
data = data[data['reputation'].notna()]

# save the data to csv
data.to_csv('csv/crypto-data-label.csv', index=False)
