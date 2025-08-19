import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv('./AirQualityUCI.csv', sep=';')

print(df.head())
print(df.tail())
print(df.shape[0])

#rimuovo colonne unnamed
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

print(df.head())
print(df.tail())

#creo la colonna datetime
df["datetime"] = pd.to_datetime(df["Date"] + " " + df["Time"], format= "%d/%m/%Y %H.%M.%S", errors='coerce' )
df = df.dropna(subset=["datetime"])
df.set_index("datetime", inplace=True)

#check
print(df.head())
print(df.tail())

#feature temporali
df["hour"] = df.index.hour
df["dayofweek"] = df.index.dayofweek
df["month"] = df.index.month

#definisco cattiva qualità dell'aria con CO maggiore della media
df.dropna(subset=["CO(GT)"], inplace=True)
df["CO(GT)"] = pd.to_numeric(df["CO(GT)"], errors='coerce')
#print(df["CO(GT)"].dtype)
df["bad_air_quality"] = (df["CO(GT)"] > df["CO(GT)"].mean()).astype(int)

#feature e target
X = df[["hour", "dayofweek", "month"]]
y = df["bad_air_quality"]

#train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

# Addestramento del modello logistic regression
model = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
model.fit(X_train, y_train)

#cambio la soglia di decisione
y_proba = model.predict_proba(X_test)[:, 1]
y_pred = (y_proba > 0.3).astype(int)

print(classification_report(y_test, y_pred))

# # Addestramento del modello Random Forest
# rf = RandomForestClassifier(random_state=42)
# rf.fit(X_train, y_train)

# y_pred_rf = rf.predict(X_test)
# print(classification_report(y_test, y_pred_rf))
