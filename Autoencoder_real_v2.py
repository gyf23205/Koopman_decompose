import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, state_dim, hidden_dim): 
        super(Encoder, self).__init__()

        print('Tanh more deeper')
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),          
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )
    
    def forward(self, x):
        return self.encoder(x)

class Decoder(nn.Module):
    def __init__(self, hidden_dim, state_dim):
        super(Decoder, self).__init__()
        self.linear = nn.Linear(hidden_dim, state_dim, bias=False)

    def forward(self, x):
        return self.linear(x)
    

class KoopmanAutoencoder(nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(KoopmanAutoencoder, self).__init__()
        self.encoder = Encoder(state_dim, hidden_dim)
        self.decoder = Decoder(hidden_dim, state_dim)
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim

        # Initialize K as an identity matrix
        # self.K = torch.eye(hidden_dim+state_dim, hidden_dim+state_dim)
        # def generate_matrix_with_partial_unit_circle_eigenvalues(n, num_unit_eigvals, seed=None):
        #     """
        #     Generate a complex matrix of size (n x n) where:
        #     - `num_unit_eigvals` eigenvalues have magnitude 1 (on the unit circle)
        #     - the remaining (n - num_unit_eigvals) are random complex numbers

        #     Returns:
        #         A (torch.Tensor): A complex matrix with the specified eigenvalue properties
        #     """
        #     if seed is not None:
        #         torch.manual_seed(seed)

        #     assert 0 <= num_unit_eigvals <= n, "num_unit_eigvals must be in range [0, n]"

        #     # Step 1: Generate desired eigenvalues
        #     angles = 2 * torch.pi * torch.rand(num_unit_eigvals)  # phases on unit circle
        #     unit_eigvals = torch.exp(1j * angles)

        #     remaining_eigvals = torch.randn(n - num_unit_eigvals) + 1j * torch.randn(n - num_unit_eigvals)
        #     eigvals = torch.cat([unit_eigvals, remaining_eigvals])

        #     # Step 2: Create a random invertible matrix V
        #     real = torch.randn(n, n)
        #     imag = torch.randn(n, n)
        #     V = torch.complex(real, imag)

        #     # Ensure V is invertible (skip in-depth check for simplicity)
        #     while torch.linalg.matrix_rank(V) < n:
        #         V = torch.complex(torch.randn(n, n), torch.randn(n, n))

        #     # Step 3: Create diagonal matrix of eigenvalues
        #     Lambda = torch.diag(eigvals)

        #     # Step 4: Construct A = V Λ V⁻¹
        #     V_inv = torch.linalg.inv(V)
        #     A = V @ Lambda @ V_inv

        #     return A

        # self.K = generate_matrix_with_partial_unit_circle_eigenvalues(hidden_dim+state_dim,10)  
        # print('New KOOPMAN INITIALIZED')
        self.K = torch.randn(hidden_dim+state_dim, hidden_dim+state_dim)  
    
    def forward(self, x):

        z = self.encoder(x)  
        
        if self.K is not None:
            z_next = torch.matmul(z, self.K.T)  # Apply computed Koopman operator
        else:
            z_next = z  

        x_hat = self.decoder(z)  
        return x_hat, z, z_next
    
    def compute_koopman_operator(self, latent_X, latent_Y):
        X_pseudo_inv = torch.linalg.pinv(latent_X)  # Compute pseudo-inverse of latent_X
        self.K = torch.matmul(latent_Y.T, X_pseudo_inv.T)  # K = Y * X^+

    # def encode(self, x):
    #     with torch.no_grad():
    #         z = self.encoder(x)  
    #     return z

    # def decode(self, x):
    #     with torch.no_grad():
    #         z = self.decoder(x)  
        # return z