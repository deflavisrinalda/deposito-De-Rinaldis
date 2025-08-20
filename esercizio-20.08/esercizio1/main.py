import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('Mall_Customers.csv')

# Esplorazione dati
print(df.head())
print(df.tail())
print(df.info())
#print(df[df['CustomerID'] == 1])

# Selezione feature da utilizzare
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]
# feature standardizzate
X_scaled = StandardScaler().fit_transform(X)

# Applico il clustering K-Means
kmeans = KMeans(n_clusters=5, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# Visualizzazione dei risultati
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=df['Cluster'], cmap='viridis')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.title('Clustering dei clienti')
plt.show()

# Identificare cluster ad alto potenziale
centroids = pd.DataFrame(kmeans.cluster_centers_, columns=['Income_centroid', 'Score_centroid'])
print(centroids)