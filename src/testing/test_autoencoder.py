import os
import torch
import numpy as np
import torch.nn as nn
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
from src.config_loader import load_config
import seaborn as sns
import json
import datetime

# ----------------------------
# CONFIGURACIÓN Y DISPOSITIVO
# ----------------------------
cfg = load_config()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_type = "dense"

# Rutas
RUN_DIR = "src/models/trained/run_20251116_192436"
MODEL_PATH = "src/models/trained/run_20251116_192436/model.pth"
TEST_DIR = cfg["paths"]["processed_data_test_dir"] 

# ----------------------------
# CARGA DEL MODELO
# ----------------------------
if model_type == "dense":
    from src.models.autoencoder import Autoencoder
    model = Autoencoder().to(device)
elif model_type == "conv":
    from src.models.convolutional_autoencoder import ConvAutoencoder
    model = ConvAutoencoder().to(device)
else:
    raise ValueError("model_type debe ser 'dense' o 'conv'")
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# ----------------------------
# FUNCIÓN PARA CALCULAR ERROR
# ----------------------------
criterion = nn.MSELoss(reduction='mean')

def reconstruction_error(x, x_hat):
    return criterion(x_hat, x).item()

# ----------------------------
# TEST LOOP
# ----------------------------
errors = []
labels = []

for file in os.listdir(TEST_DIR):
    if not file.endswith(".npy"):
        continue

    x = np.load(os.path.join(TEST_DIR, file))
    x = torch.tensor(x, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, H, W)
    
    with torch.no_grad():
        x_hat = model(x)
        err = reconstruction_error(x, x_hat)

    errors.append(err)
    labels.append(1 if "anomaly" in file.lower() else 0)  # 1 = anómalo, 0 = normal

errors = np.array(errors)
labels = np.array(labels)

# ----------------------------
# MÉTRICAS
# ----------------------------
roc_auc = roc_auc_score(labels, errors)

# threshold óptimo
fpr, tpr, thresholds = roc_curve(labels, errors)
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]

preds = (errors >= optimal_threshold).astype(int)

precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
cm = confusion_matrix(labels, preds)

# ----------------------------
# RESULTADOS
# ----------------------------
print("Evaluation Results")
print("-----------------------------")
print(f"ROC AUC: {roc_auc:.4f}")
print(f"Optimal Threshold: {optimal_threshold:.6f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-score:  {f1:.4f}")
print("Confusion Matrix:")
print(cm)

# Guardar métricas numéricas
results = {
    "timestamp": str(datetime.datetime.now()),
    "model_type": model_type,
    "roc_auc": float(roc_auc),
    "precision": float(precision),
    "recall": float(recall),
    "f1": float(f1),
    "threshold": float(optimal_threshold),
    "confusion_matrix": cm.tolist(),
    "num_test_samples": len(labels),
    "num_anomalies": int(sum(labels)),
    "num_normals": int(len(labels) - sum(labels))
}

with open(os.path.join(RUN_DIR, "test_results.json"), "w") as f:
    json.dump(results, f, indent=4)

print(f"\n Test metrics saved to {RUN_DIR}/test_results.json")

# Guardar el gráfico del histograma
plt.figure(figsize=(7,5))
plt.hist(errors[labels==0], bins=30, alpha=0.7, label="Normal")
plt.hist(errors[labels==1], bins=30, alpha=0.7, label="Anomaly")
plt.axvline(optimal_threshold, color='red', linestyle='--', label=f"Threshold={optimal_threshold:.4f}")
plt.xlabel("Reconstruction Error")
plt.ylabel("Count")
plt.legend()
plt.title("Error Distribution - Normal vs Anomaly")
plt.tight_layout()
plt.savefig(os.path.join(RUN_DIR, "error_histogram.png"))
plt.close()

# Guardar matriz de confusión como imagen
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Pred Normal', 'Pred Anomaly'],
            yticklabels=['True Normal', 'True Anomaly'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.savefig(os.path.join(RUN_DIR, "confusion_matrix.png"))
plt.close()

print(f" Plots saved in {RUN_DIR}")