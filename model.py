import pandas as pd
from sklearn.cluster import KMeans

def train_model(df, n_clusters=5):
    X = df[['Annual Income', 'Spending Score']]
    model = KMeans(n_clusters=n_clusters, random_state=42)
    df['Cluster'] = model.fit_predict(X)
    return model, df
