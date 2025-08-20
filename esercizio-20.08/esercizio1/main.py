import pandas as pd

df = pd.read_csv('Mall_Customers.csv')

# Esplorazione dati
print(df.head())
print(df.tail())
print(df.info())
print(df[df['CustomerID'] == 1])