# K-Means Cluster Analysis

This project performs a clustering analysis using the PyCaret machine learning library. The goal is to identify patterns in shipment quantity and frequency across different genera of plants and locations.

## Cloning the Repository

To clone the repository, follow the [link](https://github.com/elisadalemans/BIAproject.git).    

## What Does This Script Do?

- Loads and cleans a dataset of shipment records, genera, and locations.
- Aggregates shipment data by GenusId or LocationId.
- Normalizes the data using robust scaling.
- Uses PyCaret to create and evaluate a KMeans clustering model.
- Prompts the user to cluster based on location or genus.
- Displays and saves an elbow plot to help choose the optimal number of clusters.
- Prompts the user to input the number of clusters.
- Assigns each record to a cluster and saves the results.
- Visualizes clusters using Seaborn's pairplot.

## Requirements

Make sure to have a version of python 3.11 or lower installed. 

Install the necessary packages by running: 

pip install -r requirements.txt

## Running the script 

1. Place your cleaned dataset in the appropriate location (default: ~/BIAproject/Dataset_csv_format/data_merged.csv).
2. Run the script: 
python3 clustering.py
3. When prompted input "location" or "genus" to cluster on either of those. 
4. When prompted, check the elbow plot and input the number of clusters. 

## Outputs

- elbow.png: the elbow plot helps to choose the correct number of clusters. 
- clusters.csv: the output file with the clusters.
- PCA plot in the browser to visualize the clusters.
- pairplot.png: pairplot of the clusters. 

## Notes
- The script uses robust scaling to reduce the influence of outliers.
- The script uses k-means clustering by default.
- Timeperiod is ignored in this analysis

