import numpy as np
import scipy.io
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
import time

#from part3 import

mat_data = scipy.io.loadmat('PIE.mat')
data = mat_data['Data']
labels = mat_data['Label'].ravel() 

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data, labels, train_size=10, random_state=42)

# PCA implementation
def my_pca(X, n_components):
    # Standardize the data
    mean_vec = np.mean(X, axis=0)
    standardized_X = X - mean_vec

    # Calculate the covariance matrix
    cov_matrix = np.cov(standardized_X, rowvar=False)

    # Calculate eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    # Sort eigenvalues and corresponding eigenvectors in descending order
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Select the top n_components eigenvectors
    top_eigenvectors = eigenvectors[:, :n_components]

    # Project the data onto the new subspace
    projected_X = standardized_X.dot(top_eigenvectors)

    return projected_X

start_time = time.time()

# Apply PCA to the training and test sets
n_components_pca = 10
X_train_pca = my_pca(X_train, n_components_pca)
X_test_pca = my_pca(X_test, n_components_pca)


# Nearest Neighbor Classifier
knn_classifier = KNeighborsClassifier(n_neighbors=5)
knn_classifier.fit(X_train_pca, y_train)

# Make predictions
y_pred = knn_classifier.predict(X_test_pca)

total_time_pca = time.time() - start_time

# Evaluate the performance
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy PCA: {accuracy * 100:.2f}%')
print(f'Running Time: {total_time_pca:.4f} seconds')


