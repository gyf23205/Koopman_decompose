import torch
import torch.nn as nn
   
# Autoencoders 
class Encoder(nn.Module):
    def __init__(self, state_dim, hidden_dim): 
        super(Encoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )
    
    def forward(self, x):
        output_nn = self.encoder(x)
        output = torch.cat([x, output_nn], dim=-1)
        return output

class Decoder(nn.Module):
    def __init__(self, hidden_dim, state_dim):
        super(Decoder, self).__init__()
        self.state_dim = state_dim
        self.decoder = nn.Sequential(
            nn.Identity(state_dim)
        )
    
    def forward(self, z):
        decoded = []
        for t in range(z.size(0)):
            decoded.append(self.decoder(z[t, 0:self.state_dim]))  # Decode at each time step
        # decoded.append(z[0:self.state_dim])
        decoded = torch.stack(decoded, dim=0)  # Stack along time_step dimension
        return decoded
    
class KoopmanAutoencoder(nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(KoopmanAutoencoder, self).__init__()
        self.encoder = Encoder(state_dim, hidden_dim)
        self.decoder = Decoder(hidden_dim, state_dim)
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim

        # Initialize K as an identity matrix
        self.K = torch.eye(hidden_dim+state_dim, hidden_dim+state_dim)  
    
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

    def encode(self, x):
        with torch.no_grad():
            z = self.encoder(x)  
        return z

    def decode(self, x):
        with torch.no_grad():
            z = self.decoder(x)  
        return z
