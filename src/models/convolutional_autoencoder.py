import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvAutoencoder(nn.Module):
    def __init__(self, latent_dim=128):
        super(ConvAutoencoder, self).__init__()

        # --- ENCODER ---
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),   # (16, 32, ~626)
            nn.ReLU(True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),  # (32, 16, ~313)
            nn.ReLU(True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # (64, 8, ~157)
            nn.ReLU(True)
        )

        # Flatten/FC
        self.flatten_dim = 64 * 8 * 157
        self.fc1 = nn.Linear(self.flatten_dim, latent_dim)
        self.fc2 = nn.Linear(latent_dim, self.flatten_dim)

        # --- DECODER ---
        # Ajustamos output_padding en ambas dimensiones
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=(0,1)),  # (~16, ~314)
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=(1,1)),  # (~32, ~628)
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=(1,1)),   # (~64, ~1256)
            nn.Sigmoid()
        )

    def forward(self, x):
        B, C, H, W = x.shape  # (batch, 1, 64, 1251)

        # --- Padding temporal (horizontal) ---
        pad_W = (0, (8 - (W % 8)) % 8)
        if pad_W[1] > 0:
            x = F.pad(x, pad=(pad_W[0], pad_W[1], 0, 0), mode="reflect")

        # --- Encoder ---
        z = self.encoder(x)
        z = torch.flatten(z, start_dim=1)
        z = self.fc1(z)
        z = self.fc2(z)
        z = z.view(-1, 64, 8, 157)

        # --- Decoder ---
        x_hat = self.decoder(z)

        # --- Recorte temporal (horizontal) ---
        if pad_W[1] > 0:
            x_hat = x_hat[..., :W]

        # --- Recorte o ajuste vertical (por si llega 1–2 px de diferencia) ---
        if x_hat.shape[2] != H:
            diff = x_hat.shape[2] - H
            if diff > 0:
                x_hat = x_hat[:, :, :-diff, :]  # recorta sobrante
            else:
                pad_top = 0
                pad_bottom = abs(diff)
                x_hat = F.pad(x_hat, (0, 0, pad_top, pad_bottom))

        return x_hat
