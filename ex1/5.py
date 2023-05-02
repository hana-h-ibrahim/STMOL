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
