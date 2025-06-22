import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from pycaret.clustering import ClusteringExperiment

# data = pd.read_csv('~/BIAproject/Dataset_csv_format/data_merged.csv')
data = pd.read_csv('/mnt/c/Users/elisa/Documents/school/BIA/project/Dataset_csv_format/data_merged.csv')

# Cleaning the data
data = data.dropna(subset=["TimePeriod"])
data = data[pd.to_numeric(data["Sum of Quantity Shipped"], errors="coerce") > 0]

# Initiate experiment
s = ClusteringExperiment()

s.setup(data,
        normalize=True, 
        normalize_method="robust",
        session_id=123,
        numeric_features=["Sum of Quantity Shipped"],
        ignore_features=["Code", "TimePeriod"])

# Create an initial model
kmeans = s.create_model('kmeans')

# Elbow plot
s.plot_model(kmeans, plot = 'elbow', save=True)

# Input the number of clusters
kmeans = s.create_model('kmeans', num_clusters = 4)

# s.evaluate_model(kmeans)
df_kmeans = s.assign_model(kmeans)
df_kmeans.head()

sns.pairplot(data=df_kmeans, hue="Cluster")
plt.show()