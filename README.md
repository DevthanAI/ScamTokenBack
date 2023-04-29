# ScamTokenBack

Detects whether a token in the Web3 environment is a scam and is likely to be deprecated or not.

## Set up virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Folders

The folders are organized as follows:

- csv: contains the csv files that are used to train the model, labeled data for testing, and the results of the model.
- img: contains the visual representation of the results.
- model: contains the code for model training.
- pickle: contains the pickle files that are used to store the trained model.
- test: contains the code for testing the model.

## Explanation

This project uses clustering to detect whether a token is a scam or not. The idea is that if a token is a scam, it will have a similar transaction history to other scam tokens. This is because the scam tokens are likely to be created by the same people, and therefore will have similar transaction histories.

Initally, K-means and HDBSCAN were used to cluster the tokens. HDBSCAN did not perform well without PCA, likely because not all the dimensions were relevant. After PCA was applied to HDBSCAN to reduce the dimensions to 2, HDBSCAN still did not perform well as it clustered all the tokens into one cluster and labeled the rest as noise. This is likely because the data is not very dense, and HDBSCAN is not suitable for sparse data. Therefore, K-means was used instead.

In conclusion, K-means performed better after PCA was used to reduce the dimensionality of the data to 2.

## Results

Afterwards, using the completed K-means model, clustering was applied to test how it would analyze tokens that had "OK" reputation (according to Etherscan), and tokens that had suspicious transaction histories. The results are as follows:

The tokens that had "OK" reputation were all in cluster 1,2,4.
The tokens that had suspicious transaction histories were mostly in 0,2,3.

Therefore, it is safe to say that if a token is found in cluster 1 and 4, it is likely to be safe.
On the other hand, if it is found in cluster 0 and 3, it is likely to be a scam.
For cluster 2, it is hard to tell, as it contains both safe and scam tokens, but it is likely to be a scam so buyers should be aware.
