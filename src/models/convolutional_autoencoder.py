import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvAutoencoder(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()

        # --- ENCODER ---
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),   # (16, 32, 157)
            nn.ReLU(True),

            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # (32, 16, 79)
            nn.ReLU(True),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # (64, 8, 40)
            nn.ReLU(True)
        )

        # Tamaño exacto después de los conv
        self.flatten_dim = 64 * 8 * 40

        self.fc1 = nn.Linear(self.flatten_dim, latent_dim)
        self.fc2 = nn.Linear(latent_dim, self.flatten_dim)

        # --- DECODER ---
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                64, 32, kernel_size=3, stride=2, padding=1,
                output_padding=(1, 1)              # → (32, 16, 79)
            ),
            nn.ReLU(True),

            nn.ConvTranspose2d(
                32, 16, kernel_size=3, stride=2, padding=1,
                output_padding=(1, 1)              # → (16, 32, 157)
            ),
            nn.ReLU(True),

            nn.ConvTranspose2d(
                16, 1, kernel_size=3, stride=2, padding=1,
                output_padding=(0, 0)              # → (1, 64, ~313)
            ),
            nn.Sigmoid()
        )

    def forward(self, x):
        B, C, H, W = x.shape   #  (B, 1, 64, 313)

        # ===== ENCODER =====
        z = self.encoder(x)
        z = z.reshape(B, -1)

        # FC → Latent
        z = self.fc1(z)
        z = self.fc2(z)

        # Reshape
        z = z.view(B, 64, 8, 40)

        # ===== DECODER =====
        x_hat = self.decoder(z)

        # Recorte horizontal (W dimension)
        if x_hat.shape[3] > W:
            x_hat = x_hat[:, :, :, :W]
        elif x_hat.shape[3] < W:
            diff = W - x_hat.shape[3]
            x_hat = F.pad(x_hat, (0, diff, 0, 0))

        # Recorte vertical (H dimension)
        if x_hat.shape[2] > H:
            x_hat = x_hat[:, :, :H, :]
        elif x_hat.shape[2] < H:
            diff = H - x_hat.shape[2]
            x_hat = F.pad(x_hat, (0, 0, 0, diff))

        return x_hat
