import pandas as pd

data1 = pd.read_csv('/mnt/c/Users/elisa/Documents/school/BIA/project/Dataset_csv_format/data_1.csv')
data2 = pd.read_csv('/mnt/c/Users/elisa/Documents/school/BIA/project/Dataset_csv_format/data_2.csv')

data = pd.concat([data1, data2], ignore_index=True)
data.to_csv('data_merged.csv', index=False)

