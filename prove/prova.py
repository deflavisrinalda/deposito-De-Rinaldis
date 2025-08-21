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

# import matplotlib.pyplot as plt
# import numpy as np

# # Generiamo 3 gruppi di punti (cluster)
# np.random.seed(42)
# group1 = np.random.normal(loc=[2, 2], scale=0.5, size=(50, 2))
# group2 = np.random.normal(loc=[6, 6], scale=0.5, size=(50, 2))
# group3 = np.random.normal(loc=[2, 6], scale=0.5, size=(50, 2))

# data = np.vstack((group1, group2, group3))

# # Visualizziamo i punti (senza clustering)
# plt.figure(figsize=(6, 6))
# plt.scatter(data[:, 0], data[:, 1], c='gray')
# plt.title("Clienti (senza etichette)")
# plt.xlabel("Spesa totale")
# plt.ylabel("Numero ordini")
# plt.grid(True)
# plt.show()

# from sklearn.cluster import KMeans

# # Applichiamo k-Means ai dati generati
# kmeans = KMeans(n_clusters=3, random_state=0)
# labels = kmeans.fit_predict(data)

# # Visualizzazione con i cluster trovati
# plt.figure(figsize=(6, 6))
# plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis')
# plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
#             c='red', marker='X', s=200, label='Centroidi')
# plt.title("Cluster trovati con k-Means")
# plt.xlabel("Spesa totale")
# plt.ylabel("Numero ordini")
# plt.legend()
# plt.grid(True)
# plt.show()

# import matplotlib.pyplot as plt
# from sklearn.datasets import make_blobs
# from sklearn.cluster import KMeans
# from sklearn.metrics import silhouette_score, silhouette_samples
# import numpy as np

# # Generiamo dati di esempio
# X, _ = make_blobs(n_samples=150, centers=3, cluster_std=1.0, random_state=42)

# # Applichiamo k-Means
# kmeans = KMeans(n_clusters=3, random_state=42)
# labels = kmeans.fit_predict(X)

# # Calcolo dei silhouette score
# silhouette_avg = silhouette_score(X, labels)
# sample_silhouette_values = silhouette_samples(X, labels)

# # Preparo i dati ordinati per cluster e silhouette
# sorted_labels = np.argsort(labels)
# sorted_scores = sample_silhouette_values[sorted_labels]
# sorted_clusters = labels[sorted_labels]

# # Colori assegnati ai cluster per chiarezza
# cluster_colors = {0: "#1f77b4", 1: "#ff7f0e", 2: "#2ca02c"}
# bar_colors = [cluster_colors[c] for c in sorted_clusters]

# # Grafico semplificato e molto leggibile
# plt.figure(figsize=(12, 4))
# plt.bar(range(len(X)), sorted_scores, color=bar_colors, edgecolor='black')
# plt.axhline(silhouette_avg, color='red', linestyle='--', linewidth=2, label=f'Media silhouette = {silhouette_avg:.2f}')
# plt.title("Silhouette Score per ogni punto (colorato per cluster)", fontsize=14)
# plt.xlabel("Punti ordinati per cluster", fontsize=12)
# plt.ylabel("Silhouette Score", fontsize=12)
# plt.xticks([])
# plt.legend()
# plt.tight_layout()
# plt.show()

# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import StandardScaler
# from sklearn.cluster import KMeans
# from sklearn.metrics import silhouette_score
# import matplotlib.pyplot as plt
# import seaborn as sns

# # Simuliamo un dataset Olist semplificato per l'esercizio
# np.random.seed(42)
# n_customers = 500

# olist_df = pd.DataFrame({
#     'customer_id': [f"C{i}" for i in range(n_customers)],
#     'recency_days': np.random.exponential(scale=90, size=n_customers),  # giorni dall’ultimo ordine
#     'frequency_orders': np.random.poisson(lam=3, size=n_customers),      # numero ordini
#     'monetary_total': np.random.gamma(shape=2, scale=150, size=n_customers),  # spesa totale
#     'avg_review_score': np.clip(np.random.normal(loc=4, scale=0.5, size=n_customers), 1, 5)  # recensioni
# })

# # Rimuoviamo clienti con 0 ordini (eventuali errori)
# olist_df = olist_df[olist_df['frequency_orders'] > 0]

# # Selezioniamo le feature per il clustering
# features = ['recency_days', 'frequency_orders', 'monetary_total', 'avg_review_score']
# X = olist_df[features]

# # Standardizziamo
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)

# # Proviamo diversi k e salviamo silhouette score
# silhouette_scores = []
# K = range(2, 7)

# for k in K:
#     kmeans = KMeans(n_clusters=k, random_state=42)
#     labels = kmeans.fit_predict(X_scaled)
#     score = silhouette_score(X_scaled, labels)
#     silhouette_scores.append(score)

# # Grafico silhouette score per diversi k
# plt.figure(figsize=(8, 5))
# sns.lineplot(x=list(K), y=silhouette_scores, marker='o')
# plt.title("Silhouette Score al variare di k")
# plt.xlabel("Numero di cluster (k)")
# plt.ylabel("Silhouette Score")
# plt.grid(True)
# plt.tight_layout()
# plt.show()

# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.datasets import make_moons
# from sklearn.cluster import KMeans, DBSCAN

# # Generazione di dati a forma non circolare (mezzalune) + aggiunta outlier
# from sklearn.datasets import make_moons

# # Dati principali
# X_base, _ = make_moons(n_samples=280, noise=0.1, random_state=42)

# # Outlier casuali distribuiti nello spazio
# outliers = np.random.uniform(low=-1.5, high=2.5, size=(20, 2))

# # Concatenazione
# X_with_outliers = np.vstack((X_base, outliers))

# # k-Means clustering (forzato in 2 gruppi)
# kmeans = KMeans(n_clusters=2, random_state=42)
# kmeans_labels = kmeans.fit_predict(X_with_outliers)

# # DBSCAN clustering (basato sulla densità, senza specificare k)
# dbscan = DBSCAN(eps=0.2, min_samples=5)
# dbscan_labels = dbscan.fit_predict(X_with_outliers)

# # Assegna colori ai cluster DBSCAN (grigio per outlier, colori per cluster)
# unique_labels = set(dbscan_labels)
# palette = sns.color_palette("Set2", len(unique_labels))
# color_map = {
#     label: palette[i] if label != -1 else (0.6, 0.6, 0.6)  # grigio per outlier
#     for i, label in enumerate(sorted(unique_labels))
# }
# colors_dbscan = [color_map[label] for label in dbscan_labels]

# # Visualizzazione: k-Means vs DBSCAN
# fig, axs = plt.subplots(1, 2, figsize=(12, 5))

# # k-Means
# axs[0].scatter(X_with_outliers[:, 0], X_with_outliers[:, 1], c=kmeans_labels, cmap='Set1', s=40, edgecolor='black')
# axs[0].set_title("k-Means: forza tutti i punti in 2 cluster")
# axs[0].set_xlabel("Feature 1")
# axs[0].set_ylabel("Feature 2")
# axs[0].grid(True)

# # DBSCAN
# axs[1].scatter(X_with_outliers[:, 0], X_with_outliers[:, 1], c=colors_dbscan, s=40, edgecolor='black')
# axs[1].set_title("DBSCAN: forma naturale + outlier (grigio)")
# axs[1].set_xlabel("Feature 1")
# axs[1].set_ylabel("Feature 2")
# axs[1].grid(True)

# plt.tight_layout()
# plt.show()

from transformers import GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

prompt = "Il cavallo bianco di Napoleone è famoso."
tokens = tokenizer.tokenize(prompt)
token_ids = tokenizer.encode(prompt)

print("Token:", tokens)
print("Numero di token:", len(token_ids))

