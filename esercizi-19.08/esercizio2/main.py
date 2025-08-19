import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('./AirQualityUCI.csv', sep=';')

print(df.head())
print(df.tail())
print(df.shape[0])

#rimuovo colonne unnamed
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

print(df.head())
print(df.tail())
print(df.shape[0])

#creo la colonna datetime
df["datetime"] = pd.to_datetime(df["Date"] + " " + df["Time"], format= "%d/%m/%Y %H.%M.%S", errors='coerce' )
df = df.dropna(subset=["datetime"])
df.set_index("datetime", inplace=True)

#check
print(df.head())
print(df.tail())
print(df.shape[0])