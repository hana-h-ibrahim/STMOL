modeled here: https://colab.research.google.com/#scrollTo=OjTfIfVnvFYA

4a)

import numpy as np

# Set mean and covariance matrix
mu = np.array([0, 0])
cov = np.array([[1, 0], [0, 1]])

# Generate 100 samples
samples = np.random.multivariate_normal(mu, cov, 100)

# Print the samples
print(samples)

****

4b) 
import numpy as np
import matplotlib.pyplot as plt

# Set mean and covariance matrix
mu = np.array([0, 0])
cov = np.array([[1, 0], [0, 1]])

# Generate 100 samples
samples = np.random.multivariate_normal(mu, cov, 100)

# Plot the samples
plt.scatter(samples[:,0], samples[:,1])
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Scatter Plot of 100 Samples from a 2D Gaussian Distribution')
plt.show()

***

4c) 

import numpy as np
import matplotlib.pyplot as plt

# Set mean and covariance matrix
mu = np.array([1, -1])
cov = np.array([[1, 0], [0, 1]])

# Generate 100 samples
samples = np.random.multivariate_normal(mu, cov, 100)

# Plot the samples
plt.scatter(samples[:,0], samples[:,1])
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Scatter Plot of 100 Samples from a 2D Gaussian Distribution with Mean (1,-1)')
plt.show()


4d) 
import numpy as np
import matplotlib.pyplot as plt

# Set mean and covariance matrices
mu = np.array([0, 0])
cov1 = np.array([[1, 0.5], [0.5, 1]])
cov2 = np.array([[1, -0.5], [-0.5, 1]])

# Generate 100 samples from each distribution
samples1 = np.random.multivariate_normal(mu, cov1, 100)
samples2 = np.random.multivariate_normal(mu, cov2, 100)

# Plot the samples from distribution 1
plt.subplot(121)
plt.scatter(samples1[:,0], samples1[:,1])
plt.xlabel('x1')
plt.ylabel('x2')

# Plot the samples from distribution 2
plt.subplot(122)
plt.scatter(samples2[:,0], samples2[:,1])
plt.xlabel('x1')
plt.ylabel('x2')

plt.tight_layout()
plt.show()




