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

# import spacy

# # Carica il modello per l'italiano
# nlp = spacy.load("it_core_news_sm")

# # Testo da analizzare
# testo = "Scrivi una poesia su un gatto che cammina sul tetto."

# # Analisi
# doc = nlp(testo)

# # Output analisi
# for token in doc:
#     print(f"Token: {token.text:12} | Lemma: {token.lemma_:12} | POS: {token.pos_:10} | Dipendenza: {token.dep_:10} | Head: {token.head.text}")

# ============================================================
# Classificazione "alto/basso consumo" su AEP_hourly
# Modelli: Logistic Regression, Random Forest, XGBoost
# - Pipeline per ogni modello
# - Normalizzazione (solo dove utile)
# - Train/Test split stratificato
# - Tuning iperparametri con Optuna (ROC-AUC, 5-fold)
# - Report finale su test
# ============================================================

# import warnings
# warnings.filterwarnings("ignore")

# import numpy as np
# import pandas as pd

# from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import roc_auc_score, classification_report

# from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import RandomForestClassifier
# from xgboost import XGBClassifier

# import optuna


# # ------------------------------------------------------------
# # 1) Caricamento dati e feature engineering minimale
# # ------------------------------------------------------------
# # CSV Kaggle AEP_hourly: colonne ["Datetime", "AEP_MW"]
# df = pd.read_csv("AEP_hourly.csv", parse_dates=["Datetime"])

# # Feature temporali semplici (tutte numeriche)
# df["hour"] = df["Datetime"].dt.hour
# df["dayofweek"] = df["Datetime"].dt.dayofweek
# df["month"] = df["Datetime"].dt.month

# # Target binario: 1 se consumo sopra la mediana globale
# df["target"] = (df["AEP_MW"] > df["AEP_MW"].median()).astype(int)

# # Selezione feature / target
# FEATURES = ["hour", "dayofweek", "month"]
# X = df[FEATURES].copy()
# y = df["target"].astype(int).copy()


# # ------------------------------------------------------------
# # 2) Train/Test split stratificato
# # ------------------------------------------------------------
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y,
#     test_size=0.20,
#     stratify=y,
#     random_state=42
# )

# # K-fold per valutazione coerente (shuffle + seed fisso)
# cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# # Rapporto classi per XGBoost (utile se sbilanciato)
# pos = y_train.sum()
# neg = len(y_train) - pos
# scale_pos_weight_default = (neg / pos) if pos > 0 else 1.0


# # ------------------------------------------------------------
# # 3) Utility: cross-val ROC-AUC per pipeline
# # ------------------------------------------------------------
# def mean_cv_auc(model: Pipeline, X, y, cv) -> float:
#     """Ritorna la media della ROC-AUC su K-fold stratificati."""
#     scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
#     return float(np.mean(scores))


# # ------------------------------------------------------------
# # 4) Tuning con Optuna: LOGISTIC REGRESSION
# #    - Richiede scaling (StandardScaler)
# # ------------------------------------------------------------
# def tune_logreg(n_trials: int = 25):
#     # Pipeline: scaler -> logistic regression
#     # (tutte le feature sono numeriche: niente ColumnTransformer)
#     def objective(trial):
#         # Spazio iperparametri "piccolo ma sensato"
#         C = trial.suggest_float("C", 1e-3, 1e2, log=True)
#         solver = trial.suggest_categorical("solver", ["lbfgs", "liblinear"])
#         class_weight = trial.suggest_categorical("class_weight", [None, "balanced"])
#         max_iter = trial.suggest_int("max_iter", 200, 1000, step=200)

#         pipe = Pipeline(steps=[
#             ("scaler", StandardScaler()),
#             ("clf", LogisticRegression(
#                 C=C,
#                 solver=solver,
#                 penalty="l2",
#                 class_weight=class_weight,
#                 max_iter=max_iter,
#                 random_state=42,
#                 n_jobs=None if solver == "liblinear" else -1
#             ))
#         ])

#         return mean_cv_auc(pipe, X_train, y_train, cv)

#     study = optuna.create_study(direction="maximize", study_name="logreg_auc")
#     study.optimize(objective, n_trials=n_trials)
#     return study


# # ------------------------------------------------------------
# # 5) Tuning con Optuna: RANDOM FOREST
# #    - Gli alberi non richiedono scaling
# # ------------------------------------------------------------
# def tune_rf(n_trials: int = 25):
#     def objective(trial):
#         n_estimators = trial.suggest_int("n_estimators", 200, 800, step=100)
#         max_depth = trial.suggest_int("max_depth", 4, 18)
#         min_samples_split = trial.suggest_int("min_samples_split", 2, 20)
#         min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 10)
#         max_features = trial.suggest_categorical("max_features", ["sqrt", "log2", None])
#         class_weight = trial.suggest_categorical("class_weight", [None, "balanced", "balanced_subsample"])

#         pipe = Pipeline(steps=[
#             ("clf", RandomForestClassifier(
#                 n_estimators=n_estimators,
#                 max_depth=max_depth,
#                 min_samples_split=min_samples_split,
#                 min_samples_leaf=min_samples_leaf,
#                 max_features=max_features,
#                 class_weight=class_weight,
#                 n_jobs=-1,
#                 random_state=42
#             ))
#         ])

#         return mean_cv_auc(pipe, X_train, y_train, cv)

#     study = optuna.create_study(direction="maximize", study_name="rf_auc")
#     study.optimize(objective, n_trials=n_trials)
#     return study


# # ------------------------------------------------------------
# # 6) Tuning con Optuna: XGBOOST
# #    - Niente scaling; tree_method 'hist' per velocità su CPU
# # ------------------------------------------------------------
# def tune_xgb(n_trials: int = 30):
#     def objective(trial):
#         params = {
#             "n_estimators": trial.suggest_int("n_estimators", 300, 900, step=100),
#             "max_depth": trial.suggest_int("max_depth", 3, 10),
#             "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.2, log=True),
#             "subsample": trial.suggest_float("subsample", 0.7, 1.0),
#             "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
#             "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 8.0),
#             "gamma": trial.suggest_float("gamma", 0.0, 3.0),
#             "reg_alpha": trial.suggest_float("reg_alpha", 1e-6, 1e-1, log=True),
#             "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 5.0, log=True),
#             # gest. sbilanciamento intorno al valore stimato
#             "scale_pos_weight": trial.suggest_float(
#                 "scale_pos_weight",
#                 max(0.5, scale_pos_weight_default * 0.5),
#                 max(1.0, scale_pos_weight_default * 1.5)
#             ),
#             # fissi
#             "objective": "binary:logistic",
#             "eval_metric": "auc",
#             "tree_method": "hist",
#             "random_state": 42,
#             "n_jobs": -1
#         }

#         pipe = Pipeline(steps=[
#             ("clf", XGBClassifier(**params))
#         ])

#         return mean_cv_auc(pipe, X_train, y_train, cv)

#     study = optuna.create_study(direction="maximize", study_name="xgb_auc",
#                                 pruner=optuna.pruners.MedianPruner(n_warmup_steps=8))
#     study.optimize(objective, n_trials=n_trials)
#     return study


# # ------------------------------------------------------------
# # 7) Esegui tuning per tutti i modelli
# # ------------------------------------------------------------
# print(">> Tuning Logistic Regression...")
# study_lr = tune_logreg(n_trials=25)
# print("Best AUC (LogReg CV):", round(study_lr.best_value, 4))
# print("Best params (LogReg):", study_lr.best_params)

# print("\n>> Tuning Random Forest...")
# study_rf = tune_rf(n_trials=25)
# print("Best AUC (RF CV):", round(study_rf.best_value, 4))
# print("Best params (RF):", study_rf.best_params)

# print("\n>> Tuning XGBoost...")
# study_xgb = tune_xgb(n_trials=30)
# print("Best AUC (XGB CV):", round(study_xgb.best_value, 4))
# print("Best params (XGB):", study_xgb.best_params)


# # ------------------------------------------------------------
# # 8) Fit finale sul TRAIN e valutazione su TEST
# #    - Usa i migliori iperparametri trovati da Optuna
# # ------------------------------------------------------------
# def fit_and_report(model_name: str, best_params: dict):
#     """Crea la pipeline finale, fa fit su TRAIN e valuta su TEST."""
#     if model_name == "logreg":
#         pipe = Pipeline(steps=[
#             ("scaler", StandardScaler()),
#             ("clf", LogisticRegression(
#                 C=best_params["C"],
#                 solver=best_params["solver"],
#                 penalty="l2",
#                 class_weight=best_params["class_weight"],
#                 max_iter=best_params["max_iter"],
#                 random_state=42,
#                 n_jobs=None if best_params["solver"] == "liblinear" else -1
#             ))
#         ])

#     elif model_name == "rf":
#         pipe = Pipeline(steps=[
#             ("clf", RandomForestClassifier(
#                 n_estimators=best_params["n_estimators"],
#                 max_depth=best_params["max_depth"],
#                 min_samples_split=best_params["min_samples_split"],
#                 min_samples_leaf=best_params["min_samples_leaf"],
#                 max_features=best_params["max_features"],
#                 class_weight=best_params["class_weight"],
#                 n_jobs=-1,
#                 random_state=42
#             ))
#         ])

#     elif model_name == "xgb":
#         # Inseriamo i parametri scelti da Optuna e fissiamo quelli di base
#         xgb_params = best_params.copy()
#         xgb_params.update({
#             "objective": "binary:logistic",
#             "eval_metric": "auc",
#             "tree_method": "hist",
#             "random_state": 42,
#             "n_jobs": -1
#         })
#         pipe = Pipeline(steps=[
#             ("clf", XGBClassifier(**xgb_params))
#         ])

#     else:
#         raise ValueError("Modello non riconosciuto")

#     # Fit su TRAIN
#     pipe.fit(X_train, y_train)

#     # Valutazione su TEST (ROC-AUC su probabilità, report su classi 0/1 con soglia 0.5)
#     proba = pipe.predict_proba(X_test)[:, 1]
#     preds = (proba >= 0.5).astype(int)

#     auc = roc_auc_score(y_test, proba)
#     print(f"\n==== {model_name.upper()} | Test ROC-AUC: {auc:.4f}")
#     print(classification_report(y_test, preds, digits=3))

#     return pipe


# final_lr = fit_and_report("logreg", study_lr.best_params)
# final_rf = fit_and_report("rf", study_rf.best_params)
# final_xgb = fit_and_report("xgb", study_xgb.best_params)

# (Opzionale) salva il modello migliore con joblib
# import joblib
# joblib.dump(final_xgb, "best_model.joblib")


from sentence_transformers import SentenceTransformer
import numpy as np

# 1) Inizializza il modello (solo CPU) - veloce e multilingue
# Buon compromesso qualità/velocità per IT: paraphrase-multilingual-MiniLM-L12-v2
model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
model = SentenceTransformer(model_name, device="cpu")

# 2) Frasi da confrontare
sentence_1 = "Eventi in montagna nel 2025"
sentence_2 = "Cosa fare sulle Alpi"

# 3) Similarità coseno con embedding normalizzati (più stabile)
def semantic_similarity(sent1: str, sent2: str, model: SentenceTransformer) -> float:
    emb = model.encode([sent1, sent2], convert_to_numpy=True, normalize_embeddings=True)
    # Con vettori unitari il coseno è il prodotto scalare
    return float(np.dot(emb[0], emb[1]))

# 4) Calcolo e stampa del risultato
score = semantic_similarity(sentence_1, sentence_2, model)
print(f"Similarità semantica tra le frasi: {score:.4f}")
