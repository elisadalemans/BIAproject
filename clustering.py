import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from pycaret.clustering import ClusteringExperiment


if __name__ == "__main__":

        # Read in the csv file
        df = pd.read_csv('~/BIAproject/Dataset_csv_format/data_merged.csv')
        # data = pd.read_csv('/mnt/c/Users/elisa/Documents/school/BIA/project/Dataset_csv_format/data_merged.csv')

        # Cleaning the data
        df = df.dropna(subset=["TimePeriod"])
        df = df[pd.to_numeric(df["Sum of Quantity Shipped"], errors="coerce") > 0]

        data = df.groupby("GenusId").agg(
        total_quantity=("Sum of Quantity Shipped", "sum"),
        avg_quantity=("Sum of Quantity Shipped", "mean"),
        std_quantity=("Sum of Quantity Shipped", "std"),
        shipment_count=("Sum of Quantity Shipped", "count"),
        location_count=("LocationId", "nunique"),
        period_count=("TimePeriod", "nunique")
        ).fillna(0).reset_index()

        # Initiate experiment
        s = ClusteringExperiment()

        s.setup(data,
                normalize=True, 
                normalize_method="robust",
                session_id=123,
                ignore_features=["GenusID"])

        # Create an initial model
        kmeans = s.create_model('kmeans')

        # Elbow plot
        s.plot_model(kmeans, plot = 'elbow', save=True)
        s.plot_model(kmeans, plot='silhouette', save=True)

        print("Please choose the number of clusters by checking the elbow plot")

        while True: 
                cluster_input = input("Please choose a number: \n")
                if cluster_input.isnumeric():
                        cluster_input = int(cluster_input)
                        break
                else:
                        print("Number out of range. Please choose between 1 and 9.\n")
               
        # Input the number of clusters
        kmeans = s.create_model('kmeans', num_clusters = cluster_input)

        df_kmeans = s.assign_model(kmeans)
        print(df_kmeans.head())
        df_kmeans.to_csv('location_clusters.csv', index=False)

        s.plot_model(kmeans, 'cluster')

        sns.pairplot(data=df_kmeans, hue="Cluster")
        plt.show()