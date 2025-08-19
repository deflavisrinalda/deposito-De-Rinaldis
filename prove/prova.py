import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('../esercizi-19.08/esercizio1/DUQ_hourly.csv', parse_dates=['Datetime'])

print(df.columns)

# Etichetta: 1 se consumo > mediana, altrimenti 0
df["target"] = (df["DUQ_MW"] > df["DUQ_MW"].median()).astype(int)

df["hour"] = df["Datetime"].dt.hour
df["dayofweek"] = df["Datetime"].dt.dayofweek
df["month"] = df["Datetime"].dt.month

# Feature: ora, giorno della settimana, mese
X = df[["hour", "dayofweek", "month"]]
y = df["target"]

# Split
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, stratify=y, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.176, stratify=y_temp, random_state=42)

print(f"Train: {X_train.shape}, Validation: {X_val.shape}, Test: {X_test.shape}")
