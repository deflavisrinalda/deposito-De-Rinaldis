import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('../esercizi-19.08/esercizio1/DUQ_hourly.csv', parse_dates=['Datetime'])

print(df.columns)

# Etichetta: 1 se consumo > mediana, altrimenti 0
df["target"] = (df["DUQ_MW"] > df["DUQ_MW"].median()).astype(int)

df["hour"] = df["Datetime"].dt.hour
df["dayofweek"] = df["Datetime"].dt.dayofweek
df["month"] = df["Datetime"].dt.month

# # Feature: ora, giorno della settimana, mese
# X = df[["hour", "dayofweek", "month"]]
# y = df["target"]

# # Split
# X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, stratify=y, random_state=42)
# X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.176, stratify=y_temp, random_state=42)

# print(f"Train: {X_train.shape}, Validation: {X_val.shape}, Test: {X_test.shape}")


from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, make_scorer

# Feature e target (come prima)
X = df[["hour", "dayofweek", "month"]]
y = df["target"]

# K-Fold stratificato
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Decision Tree
tree = DecisionTreeClassifier(max_depth=5, random_state=42)
auc_tree = cross_val_score(tree, X, y, cv=skf, scoring="roc_auc")

# Neural Network con scaling
mlp_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("mlp", MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300, random_state=42))
])
auc_mlp = cross_val_score(mlp_pipeline, X, y, cv=skf, scoring="roc_auc")

print(f"Decision Tree AUC: {auc_tree.mean():.3f} ± {auc_tree.std():.3f}")
print(f"Neural Network AUC: {auc_mlp.mean():.3f} ± {auc_mlp.std():.3f}")

