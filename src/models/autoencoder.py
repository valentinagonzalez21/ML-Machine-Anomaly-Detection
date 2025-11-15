import torch
import torch.nn as nn # neural networks - submódulo de pytorch 

# Definición de módulo de pytorch

class Autoencoder(nn.Module):

    # Parámetros de entrada:
    # input_dim: número total de features de entrada -- tamaño del espectrograma
    # latent_dim: tamaño del espacio latente (cuanto más chico más compresión)
    def __init__(self, input_dim=64*313, latent_dim=128): 
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 1024), # aprende transformación para 1024 neuronas
            nn.ReLU(), # no linealidad (aplana valores negativos a 0 y deja el resto igual)
            nn.Linear(1024, latent_dim), # bottleneck (la compresión) -- salida 
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, input_dim),
            nn.Sigmoid() # comprime la salida a [0, 1]
        )

    # cómo fluye un batch por la red
    def forward(self, x):
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon.view(x.size(0), 1, 64, 313)
