# import pandas as pd
# from sklearn.model_selection import train_test_split

# df = pd.read_csv('../esercizi-19.08/esercizio1/DUQ_hourly.csv', parse_dates=['Datetime'])

# print(df.columns)

# # Etichetta: 1 se consumo > mediana, altrimenti 0
# df["target"] = (df["DUQ_MW"] > df["DUQ_MW"].median()).astype(int)

# df["hour"] = df["Datetime"].dt.hour
# df["dayofweek"] = df["Datetime"].dt.dayofweek
# df["month"] = df["Datetime"].dt.month

# # Feature: ora, giorno della settimana, mese
# X = df[["hour", "dayofweek", "month"]]
# y = df["target"]

# # Split
# X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, stratify=y, random_state=42)
# X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.176, stratify=y_temp, random_state=42)

# print(f"Train: {X_train.shape}, Validation: {X_val.shape}, Test: {X_test.shape}")


# from sklearn.model_selection import StratifiedKFold, cross_val_score
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.neural_network import MLPClassifier
# from sklearn.preprocessing import StandardScaler
# from sklearn.pipeline import Pipeline
# from sklearn.metrics import roc_auc_score, make_scorer

# # Feature e target (come prima)
# X = df[["hour", "dayofweek", "month"]]
# y = df["target"]

# # K-Fold stratificato
# skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# # Decision Tree
# tree = DecisionTreeClassifier(max_depth=5, random_state=42)
# auc_tree = cross_val_score(tree, X, y, cv=skf, scoring="roc_auc")

# # Neural Network con scaling
# mlp_pipeline = Pipeline([
#     ("scaler", StandardScaler()),
#     ("mlp", MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300, random_state=42))
# ])
# auc_mlp = cross_val_score(mlp_pipeline, X, y, cv=skf, scoring="roc_auc")

# print(f"Decision Tree AUC: {auc_tree.mean():.3f} ± {auc_tree.std():.3f}")
# print(f"Neural Network AUC: {auc_mlp.mean():.3f} ± {auc_mlp.std():.3f}")

import matplotlib.pyplot as plt
import numpy as np

# Generiamo 3 gruppi di punti (cluster)
np.random.seed(42)
group1 = np.random.normal(loc=[2, 2], scale=0.5, size=(50, 2))
group2 = np.random.normal(loc=[6, 6], scale=0.5, size=(50, 2))
group3 = np.random.normal(loc=[2, 6], scale=0.5, size=(50, 2))

data = np.vstack((group1, group2, group3))

# Visualizziamo i punti (senza clustering)
plt.figure(figsize=(6, 6))
plt.scatter(data[:, 0], data[:, 1], c='gray')
plt.title("Clienti (senza etichette)")
plt.xlabel("Spesa totale")
plt.ylabel("Numero ordini")
plt.grid(True)
plt.show()

from sklearn.cluster import KMeans

# Applichiamo k-Means ai dati generati
kmeans = KMeans(n_clusters=3, random_state=0)
labels = kmeans.fit_predict(data)

# Visualizzazione con i cluster trovati
plt.figure(figsize=(6, 6))
plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
            c='red', marker='X', s=200, label='Centroidi')
plt.title("Cluster trovati con k-Means")
plt.xlabel("Spesa totale")
plt.ylabel("Numero ordini")
plt.legend()
plt.grid(True)
plt.show()
