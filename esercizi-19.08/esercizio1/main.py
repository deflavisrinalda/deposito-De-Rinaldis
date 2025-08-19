import pandas as pd

file_path = './DUQ_hourly.csv'
df = pd.read_csv(file_path)

print(df.head())

print(df.tail())