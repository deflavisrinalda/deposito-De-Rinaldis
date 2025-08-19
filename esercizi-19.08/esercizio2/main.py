import pandas as pd

df = pd.read_csv('./AirQualityUCI.csv', sep=';')

print(df.head())
print(df.tail())