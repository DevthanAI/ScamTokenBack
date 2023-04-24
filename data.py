import pandas as pd
import numpy as np

# Read the data
data = pd.read_csv('crypto-data.csv')

# remove \n from Name column
data['Name'] = data['Name'].str.replace('\n', '')


