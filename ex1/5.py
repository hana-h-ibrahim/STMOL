c) 


import numpy as np
import matplotlib.pyplot as plt

# Set mean and covariance matrix
mu = np.array([0, 0])
cov = np.array([[2, 1], [1, 3]])

# Generate 100 samples
samples = np.random.multivariate_normal(mu, cov, 100)

# Plot the samples
plt.scatter(samples[:,0], samples[:,1])
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Scatter Plot of 100 Samples from a 2D Gaussian Distribution with Covariance Matrix A')
plt.show()

***

import numpy as np
import matplotlib.pyplot as plt

# Define mean and covariance matrices
mu = np.array([0, 0])

cov1 = np.array([[2, 0], [0, 1]])  # elliptical
cov2 = np.array([[1, -0.5], [-0.5, 1]])  # elongated
cov3 = np.array([[1, 0], [0, 2]])  # elliptical

# Define ranges for x1 and x2
x1_range = np.linspace(-3, 3, 100)
x2_range = np.linspace(-3, 3, 100)

# Create a grid of (x1, x2) points
x1_grid, x2_grid = np.meshgrid(x1_range, x2_range)
x_grid = np.stack((x1_grid, x2_grid), axis=-1)

# Compute the PDF for each (x1, x2) point on the grid
pdf1 = np.zeros_like(x1_grid)
pdf2 = np.zeros_like(x1_grid)
pdf3 = np.zeros_like(x1_grid)

for i in range(len(x1_range)):
    for j in range(len(x2_range)):
        x = x_grid[i, j]
        pdf1[i, j] = 1/(2*np.pi*np.sqrt(np.linalg.det(cov1))) * np.exp(-1/2 * np.dot((x-mu), np.linalg.solve(cov1, (x-mu).T)))
        pdf2[i, j] = 1/(2*np.pi*np.sqrt(np.linalg.det(cov2))) * np.exp(-1/2 * np.dot((x-mu), np.linalg.solve(cov2, (x-mu).T)))
        pdf3[i, j] = 1/(2*np.pi*np.sqrt(np.linalg.det(cov3))) * np.exp(-1/2 * np.dot((x-mu), np.linalg.solve(cov3, (x-mu).T)))

# Plot the contour plots of the PDF
plt.figure(figsize=(10, 3))

plt.subplot(131)
plt.contour(x1_grid, x2_grid, pdf1, levels=10)
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Elliptical Covariance Matrix')

plt.subplot(132)
plt.contour(x1_grid, x2_grid, pdf2, levels=10)
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Elongated Covariance Matrix')

plt.subplot(133)
plt.contour(x1_grid, x2_grid, pdf3, levels=10)
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Elliptical Covariance Matrix')

plt.tight_layout()
plt.show()

***

import numpy as np

def generate_covariance_matrix(n):
    # Generate a lower triangular matrix with positive diagonal entries
    L = np.tril(np.random.rand(n, n))
    L[np.diag_indices_from(L)] = np.exp(L[np.diag_indices_from(L)])
    
    # Compute the covariance matrix as Sigma = L*L^T
    Sigma = np.dot(L, L.T)
    return Sigma

***


import numpy as np
import matplotlib.pyplot as plt

# Define a function to generate a random vector with a given covariance matrix
def generate_random_vector(cov_matrix):
  # Get the dimension of the covariance matrix
  dim = cov_matrix.shape[0]
  # Get the eigenvalues and eigenvectors of the covariance matrix
  eigvals, eigvecs = np.linalg.eig(cov_matrix)
  # Generate a standard normal random vector of the same dimension
  z = np.random.randn(dim)
  # Transform the standard normal random vector using the eigenvectors and eigenvalues
  x = eigvecs @ np.diag(np.sqrt(eigvals)) @ z
  # Return the transformed random vector
  return x

# Define some example covariance matrices with different dimensions and eigenvalues
cov_matrix_1 = np.array([[4, 2], [2, 3]]) # 2 x 2 matrix with eigenvalues 5 and 2
cov_matrix_2 = np.array([[1, -0.5, 0], [-0.5, 2, -0.8], [0, -0.8, 3]]) # 3 x 3 matrix with eigenvalues 3.6, 1.9 and 0.5
cov_matrix_3 = np.array([[9, -6, -3], [-6, 16, -4], [-3, -4, 9]]) # 3 x 3 matrix with eigenvalues 20, 10 and 4

# Generate some random vectors with the given covariance matrices
x_1 = generate_random_vector(cov_matrix_1)
x_2 = generate_random_vector(cov_matrix_2)
x_3 = generate_random_vector(cov_matrix_3)

# Plot the histograms of the random vectors
plt.figure(figsize=(12,8))
plt.subplot(2,2,1)
plt.hist(x_1)
plt.title("Histogram of x_1 with cov_matrix_1")
plt.subplot(2,2,2)
plt.hist(x_2)
plt.title("Histogram of x_2 with cov_matrix_2")
plt.subplot(2,2,3)
plt.hist(x_3)
plt.title("Histogram of x_3 with cov_matrix_3")
plt.show()
