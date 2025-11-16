import torch
import torch.optim as optim # optimizadores
import torch.nn as nn
from torch.utils.data import DataLoader
from src.models.autoencoder import Autoencoder  
from src.dataloader import MelSpectrogramDataset
from src.config_loader import load_config
import os
import json
from datetime import datetime
import matplotlib.pyplot as plt

# Parametros
cfg = load_config()
epochs = cfg["training"]["epochs"]
batch_size = cfg["training"]["batch_size"]
learning_rate = cfg["training"]["learning_rate"]
base_dir = cfg["paths"]["trained_models_dir"] # Carpeta base para guardar modelos entrenados
os.makedirs(base_dir, exist_ok=True)

# Tipo de modelo 
model_type = "dense" # "dense" o "conv"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # si hay GPU se usa, sino CPU

if model_type == "dense":
    from src.models.autoencoder import Autoencoder
    model_name = "autoencoder"
    model = Autoencoder().to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
elif model_type == "conv":
    from src.models.convolutional_autoencoder import ConvAutoencoder
    model_name = "conv_autoencoder"
    model = ConvAutoencoder().to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
else:
    raise ValueError("model_type debe ser 'dense' o 'conv'")

print(f"Entrenando modelo '{model_name}' en: {device}")

# Dataset y dataloader
dataset = MelSpectrogramDataset() # lee los .npy y devuelve tensores
dataloader = DataLoader(dataset, batch_size, shuffle=True) # empaqueta el dataset en batches y mezcla el orden

# Modelo
criterion = nn.MSELoss() # función de pérdida
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


# Subcarpeta única con fecha y hora
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
run_dir = os.path.join(base_dir, f"run_{timestamp}")
os.makedirs(run_dir, exist_ok=True)

# Paths de guardado
model_path = os.path.join(run_dir, "model.pth")
log_path = os.path.join(run_dir, "training_log.json")
loss_plot_path = os.path.join(run_dir, "loss_curve.png")

# Entrenamiento
epoch_losses = []
for epoch in range(epochs):
    epoch_loss = 0.0
    for mel in dataloader:
        mel = mel.unsqueeze(1).to(device)  # (B,1,H,W)
        output = model(mel)
        loss = criterion(output, mel)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(dataloader)
    epoch_losses.append(avg_loss)
    print(f"Epoch [{epoch+1}/{epochs}]  Loss promedio: {avg_loss:.6f}")

# Guardar modelo
torch.save(model.state_dict(), model_path)
print(f"Modelo guardado en {model_path}")

# Guardar log JSON con info del entrenamiento
log_data = {
    "timestamp": timestamp,
    "run_dir": run_dir,
    "device": str(device),
    "model_type": model_type,
    "epochs": epochs,
    "batch_size": batch_size,
    "learning_rate": learning_rate,
    "config_file": "config.yaml",
    "final_loss": epoch_losses[-1],
    "losses": epoch_losses
}
with open(log_path, "w") as f:
    json.dump(log_data, f, indent=4)
print(f"Log guardado en: {log_path}")

# Graficar y guardar curva de pérdida
plt.figure()
plt.plot(range(1, epochs + 1), epoch_losses, marker="o")
plt.title("Curva de pérdida del entrenamiento")
plt.xlabel("Época")
plt.ylabel("Loss promedio (MSE)")
plt.grid(True)
plt.tight_layout()
plt.savefig(loss_plot_path)
plt.close()
print(f"Curva de pérdida guardada en: {loss_plot_path}")