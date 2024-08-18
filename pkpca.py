import torch
import os

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def linear_kernel(X, Y=None):
    """
    Compute the linear kernel between X and Y.

    Args:
    - X: A tensor of shape (n_samples_X, n_features).
    - Y: A tensor of shape (n_samples_Y, n_features) or None.

    Returns:
    - A tensor of shape (n_samples_X, n_samples_Y) representing the kernel matrix.
    """
    if Y is None:
        Y = X

    return torch.matmul(X, Y.T)


def center_kernel(K):
    """
    Center the kernel matrix K.

    Args:
    - K: A tensor of shape (n_samples, n_samples).

    Returns:
    - The centered kernel matrix.
    """
    n_samples = K.shape[0]
    one_n = torch.ones((n_samples, n_samples), device=K.device) / n_samples
    K_centered = K - one_n @ K - K @ one_n + one_n @ K @ one_n
    return K_centered


class ProbabilisticKernelPCA:
    def __init__(self, n_components, kernel=linear_kernel):
        self.n_components = n_components
        self.kernel = kernel

    def fit(self, X):
        """
        Fit the model with X.

        Args:
        - X: A tensor of shape (n_samples, n_features).
        """
        self.X_fit_ = X.to(device)
        K = self.kernel(self.X_fit_)
        K_centered = center_kernel(K)

        # Transfer to CPU for eigen-decomposition
        K_centered_cpu = K_centered.to("cpu")
        eigenvalues, eigenvectors = torch.linalg.eigh(K_centered_cpu)

        # Transfer results back to MPS device
        eigenvalues = eigenvalues.to(device)
        eigenvectors = eigenvectors.to(device)

        # Sort eigenvalues and eigenvectors
        idx = torch.argsort(eigenvalues, descending=True)
        self.eigenvalues_ = eigenvalues[idx]
        self.eigenvectors_ = eigenvectors[:, idx]

        # Select the top n_components
        self.alphas_ = self.eigenvectors_[:, :self.n_components]
        self.lambdas_ = self.eigenvalues_[:self.n_components]

    def transform(self, X):
        """
        Apply the kernel PCA transformation to X.

        Args:
        - X: A tensor of shape (n_samples, n_features).

        Returns:
        - X_transformed: A tensor of shape (n_samples, n_components).
        """
        K = self.kernel(X.to(device), self.X_fit_)
        K_centered = K - torch.mean(K, dim=1, keepdim=True) - torch.mean(self.kernel(self.X_fit_), dim=0,
                                                                         keepdim=True) + torch.mean(
            self.kernel(self.X_fit_))
        X_transformed = K_centered @ self.alphas_ / torch.sqrt(self.lambdas_)
        return X_transformed

    def fit_transform(self, X):
        """
        Fit the model with X and apply the kernel PCA transformation.

        Args:
        - X: A tensor of shape (n_samples, n_features).

        Returns:
        - X_transformed: A tensor of shape (n_samples, n_components).
        """
        self.fit(X)
        return self.transform(X)


def pkPCA(X, n_iter=100, tol=10e-4):
    X_centered = X - torch.mean(X, dim=0, keepdim=True)
    # Initialize parameters
    n_samples, n_features = X_centered.size()
    latent_dim = min([opt.h_dim, n_samples])
    W = torch.randn(n_features, latent_dim).to(X_centered.device)
    sigma2 = 0.1

    # Initialize latent variables
    h = torch.randn(n_samples, latent_dim)

    for _ in range(n_iter):
        # E-step: Compute expected values of latent variables
        sigma2_I = sigma2 * torch.eye(latent_dim).to(X_centered.device)
        M = (torch.mm(W.T, W) + sigma2_I).cpu()
        W_T_X = torch.mm(X_centered, W).cpu()
        h = torch.linalg.solve(M, W_T_X.T).T.to(X_centered.device)
        # M-step: Update parameters
        M_inv = torch.linalg.inv(M).to(X_centered.device)
        W = torch.mm(X_centered.T, h).mm(M_inv)
        sigma2 = torch.mean(torch.sum((X_centered - torch.mm(h, W.T)) ** 2, dim=1)) / n_features

        # Check for convergence
        if torch.norm(X_centered - torch.mm(h, W.T)) / n_samples < tol:
            break

    a = torch.mm(X, torch.t(X))
    nh1 = a.size(0)
    oneN = torch.div(torch.ones(nh1, nh1), nh1).float().to(opt.device)
    a = a - torch.mm(oneN, a) - torch.mm(a, oneN) + torch.mm(torch.mm(oneN, a), oneN)  # centering
    _, s, _ = torch.linalg.svd(a, full_matrices=False)
    return h[:, :opt.h_dim], s



# Example usage
if __name__ == "__main__":
    # Create a sample dataset
    from sklearn.datasets import make_moons
    import matplotlib.pyplot as plt

    X, y = make_moons(n_samples=100, noise=0.1)
    X = torch.tensor(X, dtype=torch.float32).to(device)

    # Apply Probabilistic Kernel PCA with a linear kernel
    pkpca = ProbabilisticKernelPCA(n_components=2)
    X_transformed = pkpca.fit_transform(X)

    # Plot the results
    plt.scatter(X_transformed[:, 0].cpu().numpy(), X_transformed[:, 1].cpu().numpy(), c=y)
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.title('Probabilistic Kernel PCA with Linear Kernel')
    plt.show()
