import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

df = pd.read_csv('Mall_Customers.csv')

# Esplorazione dati
print(df.head())
print(df.tail())
print(df.info())
print(df[df['CustomerID'] == 1])

# Selezione feature da utilizzare
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

# Applico il clustering K-Means
kmeans = KMeans(n_clusters=5, random_state=42)
df['Cluster'] = kmeans.fit_predict(X)

# Visualizzazione dei risultati
plt.scatter(X['Annual Income (k$)'], X['Spending Score (1-100)'], c=df['Cluster'], cmap='viridis')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.title('Clustering dei clienti')
plt.show()