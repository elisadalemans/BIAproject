import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from pycaret.clustering import ClusteringExperiment

def load_data(filepath):
        """
        Loads and cleans the data.
        """
        df = pd.read_csv(filepath)

        # Cleaning the data
        df = df.dropna(subset=["TimePeriod"])
        df = df[pd.to_numeric(df["Sum of Quantity Shipped"], errors="coerce") > 0]
        return df

def choose_cluster():
        """
        Get input from the user.
        Asks whether to cluster based on Location or Genus.
        """
        while True:
                choice = input("Cluster by [location/genus]? ").strip().lower()
                if choice in ["location", "genus"]:
                        return choice
                print("Invalid input. Please enter 'location' or 'genus'.")
                        
def aggregate_data(df, cluster_by):
        """
        Aggregates data based on LocationId or GenusId
        """
        if cluster_by == "location":
                cluster = "LocationId"
                ignore = "GenusId"
        else:
                cluster = "GenusId"
                ignore = "LocationId"

        data = df.groupby(cluster).agg(
                total_quantity=("Sum of Quantity Shipped", "sum"),
                shipment_count=("Sum of Quantity Shipped", "count"),
        ).fillna(0).reset_index()
        return data, cluster, ignore

def number_clusters():
        """
        Prompts the user for the number of clusters.
        """ 
        while True: 
                cluster_input = input("Please choose a number: \n")
                if cluster_input.isnumeric():
                        return int(cluster_input)
                else:
                        print("Number out of range. Please choose between 1 and 9.\n")

def main():
        filepath = '~/BIAproject/Dataset_csv_format/data_merged.csv'

        # Load the data
        data = load_data(filepath)
        
        # Cluster based on location or genus
        cluster_choice = choose_cluster()

        # Aggregate the data
        data, cluster, ignore = aggregate_data(data, cluster_choice)

        # Initiate the experiment
        s = ClusteringExperiment()

        s.setup(data,
                normalize=True, 
                normalize_method="robust",
                session_id=123,
                ignore_features=[ignore])

         # Create an initial model
        kmeans = s.create_model('kmeans')

        # Elbow plot
        s.plot_model(kmeans, plot = 'elbow', save=True)

        print("Please choose the number of clusters by checking the elbow plot")

        num_clusters = number_clusters()
               
        # Input the number of clusters
        kmeans = s.create_model('kmeans', num_clusters = num_clusters)
        df_kmeans = s.assign_model(kmeans)

        print(df_kmeans.head())

        # Saves the output to a csv file
        df_kmeans.to_csv('clusters.csv', index=False)

        # Creates a PCA plot of the clusters
        s.plot_model(kmeans, 'cluster')

        # Creates a pairplot of the clusters
        sns.pairplot(data=df_kmeans, hue="Cluster")
        plt.show()

if __name__ == "__main__":
        main()