import pandas as pd

df = pd.read_csv('eco_foods.csv')
print(df.head(10))  # View first 10 rows
print(df.describe())  # Stats on footprints (e.g., mean ~4.0 kg)
   