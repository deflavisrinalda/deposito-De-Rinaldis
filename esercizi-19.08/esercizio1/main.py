import pandas as pd

#estraggo il file
file_path = './DUQ_hourly.csv'
df = pd.read_csv(file_path, parse_dates=['Datetime'], index_col='Datetime')

#controllo delle prime e ultime righe
print(df.head())
print(df.tail())

colonna_consumo = "DUQ_MW"

#classidicazione rispetto alla media globale
media_globale = df[colonna_consumo].mean()
df["classificazione_globale"] = df[colonna_consumo].apply(lambda x: "sotto media" if x < media_globale else "sopra media")

#check creazione classificazione globale
print(df.head())
print(df.tail())

#classificazione rispetto alla media giornaliera
media_giornaliera = df.groupby(df.index.date)[colonna_consumo].mean().mean()
df["classificazione_giornaliera"] = df[colonna_consumo].apply(lambda x: "sotto media" if x < media_giornaliera else "sopra media")

#check creazione classificazione giornaliera
print(df.head())
print(df.tail())
