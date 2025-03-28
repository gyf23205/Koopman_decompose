import numpy as np
from scipy.linalg import eig

A_dense = np.random.rand(100, 100)

# Compute both left and right eigenvectors simultaneously
eigvals, eigvecs_left, eigvecs_right = eig(A_dense, left=True, right=True)

print("Eigenvalues shape:", eigvals.shape)
print("Left eigenvectors shape:", eigvecs_left.shape)
print("Right eigenvectors shape:", eigvecs_right.shape)
