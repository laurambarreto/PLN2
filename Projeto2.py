from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.cluster import KMeans

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

KNN = KNeighborsClassifier (n_neighbors = 3) 
KNN.fit (X_train, y_train)

y_pred = KNN.predict (X_test)

print (y_pred)

print("Acurácia:", accuracy_score(y_test, y_pred))
print("\nRelatório de classificação:")
print(classification_report(y_test, y_pred, target_names=['Mentira (-1)', 'Citação (0)', 'Viés (1)']))

kmeans = KMeans (n_clusters = 3, random_state = 42)
kmeans.fit (X_train)

y_pred = kmeans.predict (X_test)