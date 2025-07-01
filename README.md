# K-Means Cluster Analysis

This project performs a clustering analysis using the PyCaret machine learning library. The goal is to identify patterns in shipment quantity and frequency across different genera of plants and locations.

## What Does This Script Do?

- Loads and cleans a dataset of shipment records, genera, and locations.
- Aggregates shipment data by GenusId or LocationId.
- Normalizes the data using robust scaling.
- Uses PyCaret to create and evaluate a KMeans clustering model.
- Displays and saves an elbow plot to help choose the optimal number of clusters.
- Prompts the user to input the number of clusters.
- Assigns each record to a cluster and saves the results.
- Visualizes clusters using Seaborn's pairplot.

## Requirements

Make sure to have a version of pyton 3.11 or lower installed. 