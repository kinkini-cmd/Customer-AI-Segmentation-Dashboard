import matplotlib.pyplot as plt


def plot_clusters(df):
    plt.figure()
    for cluster in df['Cluster'].unique():
        subset = df[df['Cluster'] == cluster]
        plt.scatter(subset['Annual Income'], subset['Spending Score'], label=f"Cluster {cluster}")

    plt.xlabel("Annual Income")
    plt.ylabel("Spending Score")
    plt.legend()
    return plt
