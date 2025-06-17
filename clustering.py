import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from pycaret.clustering import ClusteringExperiment

data = pd.read_csv('/mnt/c/Users/elisa/Documents/school/BIA/project/Dataset_csv_format/data.csv')

# Display basic information about the dataset
# print(data.info())
# print(data.describe())

# # Visualize the distribution of the first few variables
# for column in data.columns[:5]:  # Adjust the range as needed
#     plt.figure(figsize=(10, 6))
#     sns.histplot(data[column], kde=True)
#     plt.title(f'Distribution of {column}')
#     plt.savefig("plot.png")

# # Scale the data
# scaler = StandardScaler()
# scaled_data = scaler.fit_transform(data)

#initiate experiment
s = ClusteringExperiment()

s.setup(data,
        normalize=True, 
        normalize_method="robust",
        session_id=123,
        numeric_features=["Sum Of Quantity Shipped"],
        ignore_features=["Code", "TimePeriod"])

# Create an initial model
kmeans = s.create_model('kmeans')

#elbow plot
s.plot_model(kmeans, plot = 'elbow')