import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


df = pd.read_csv('~/BIAproject/location_clusters.csv')

features = ['total_quantity', 'avg_quantity', 'std_quantity', 'shipment_count', 'location_count', 'period_count']

for feature in features:
    plt.figure(figsize=(8, 4))
    sns.barplot(data=df, x='Cluster', y=feature, palette='viridis')
    plt.title(f'Mean {feature} by Cluster')
    plt.show()
