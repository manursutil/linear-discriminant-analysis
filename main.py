import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt


class LDA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.linear_discriminants = None

    def fit(self, X, y):
        # Get number of features and unique class labels
        n_features = X.shape[1]
        class_labels = np.unique(y)

        # Calculate overall mean of all data points
        mean_overall = np.mean(X, axis=0)

        # Initialize within-class and between-class scatter matrices
        S_W = np.zeros((n_features, n_features))  # Within-class scatter
        S_B = np.zeros((n_features, n_features))  # Between-class scatter

        for c in class_labels:
            # Get data points for current class
            X_c = X[y == c]
            mean_c = np.mean(X_c, axis=0)

            # Add to within-class scatter matrix
            S_W += (X_c - mean_c).T.dot(X_c - mean_c)

            # Add to between-class scatter matrix
            n_C = X_c.shape[0]
            mean_diff = (mean_c - mean_overall).reshape(n_features, 1)
            S_B += n_C * (mean_diff).dot(mean_diff.T)

        # Solve the generalized eigenvalue problem: S_W^(-1) * S_B
        A = np.linalg.inv(S_W).dot(S_B)

        # Get eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eig(A)
        eigenvectors = eigenvectors.T

        # Sort eigenvectors by eigenvalues (descending order)
        idxs = np.argsort(abs(eigenvalues))[::-1]
        eigenvalues = eigenvalues[idxs]
        eigenvectors = eigenvectors[idxs]

        # Store the top n_components eigenvectors as linear discriminants
        self.linear_discriminants = eigenvectors[0 : self.n_components]

    def transform(self, X):
        # Project data onto the linear discriminants
        return np.dot(X, self.linear_discriminants.T)


def main():
    # Load the iris dataset
    data = datasets.load_iris()
    X = data.data
    y = data.target

    # Create LDA instance and reduce to 2 dimensions
    lda = LDA(2)
    lda.fit(X, y)
    X_projected = lda.transform(X)

    # Print original and transformed data shapes
    print("Shape of X:", X.shape)
    print("Shape if transformed X:", X_projected.shape)

    # Extract the two projected dimensions for plotting
    x1 = X_projected[:, 0]
    x2 = X_projected[:, 1]

    # Create scatter plot colored by class
    plt.scatter(
        x1, x2, c=y, edgecolor="none", alpha=0.8, cmap=plt.cm.get_cmap("viridis", 3)
    )

    plt.xlabel("Linear Discriminant 1")
    plt.ylabel("Linear Discriminant 2")
    plt.colorbar()
    plt.show()


if __name__ == "__main__":
    main()
