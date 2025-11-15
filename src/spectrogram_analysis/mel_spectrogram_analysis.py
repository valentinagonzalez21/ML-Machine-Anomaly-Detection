### 🔹 BLOQUE 1 — Análisis de energía global y por banda

import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display

from src.config_loader import load_config 

cfg = load_config() # load configuration

def analyze_energy_distribution(audio_path, cfg):
    """
    Analiza la distribución de energía en el espectrograma Mel de un audio.
    Permite observar qué bandas Mel concentran más energía.
    """
    # Cargar audio con la tasa de muestreo definida en config.yaml
    y, sr = librosa.load(audio_path, sr=cfg["audio"]["sample_rate"])

    # Calcular espectrograma Mel según los parámetros de configuración
    S = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=cfg["audio"]["n_mels"],
        n_fft=cfg["audio"]["n_fft"],
        hop_length=cfg["audio"]["hop_length"],
        power=cfg["audio"]["power"]
    )

    # Convertir a escala logarítmica (dB)
    S_dB = librosa.power_to_db(S, ref=np.max)

    # Energía promedio por banda Mel
    energy_per_band = np.mean(S, axis=1)
    
    # Gráfico de energía promedio
    plt.figure(figsize=(10, 4))
    plt.plot(energy_per_band)
    plt.title("Energía promedio por banda Mel")
    plt.xlabel("Índice de banda Mel")
    plt.ylabel("Energía promedio")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.show()

    # Energía total (suma de todas las celdas del espectrograma)
    total_energy = np.sum(S)
    print(f"Energía total del espectrograma: {total_energy:.2e}")

    return S_dB, energy_per_band

def parameter_sweep(audio_path, sample_rate=None):
    """
    Compara cómo cambia la energía total del espectrograma
    al variar n_mels, n_fft y hop_length.
    """
    import librosa
    import numpy as np

    n_mels_list = [32, 64, 128]
    n_fft_list = [512, 1024, 2048]
    hop_lengths = [128, 256, 512]

    y, sr = librosa.load(audio_path, sr=sample_rate)
    results = []

    for n_mels in n_mels_list:
        for n_fft in n_fft_list:
            for hop in hop_lengths:
                S = librosa.feature.melspectrogram(
                    y=y, sr=sr, n_mels=n_mels,
                    n_fft=n_fft, hop_length=hop
                )
                total_energy = np.sum(S)
                results.append((n_mels, n_fft, hop, total_energy))

    # Ordenar por energía total
    results.sort(key=lambda x: x[3], reverse=True)
    print("Top configuraciones por energía total:")
    for n_mels, n_fft, hop, E in results[:5]:
        print(f"n_mels={n_mels}, n_fft={n_fft}, hop_length={hop}  → energía={E:.2e}")

    return results

def compare_normalization_effect(audio_path, cfg):
    """
    Muestra el efecto visual de aplicar o no normalización
    al espectrograma Mel.
    """
    y, sr = librosa.load(audio_path, sr=cfg["audio"]["sample_rate"])

    # Espectrograma sin normalizar
    S = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=cfg["audio"]["n_mels"],
        n_fft=cfg["audio"]["n_fft"],
        hop_length=cfg["audio"]["hop_length"],
        power=cfg["audio"]["power"]
    )
    S_db = librosa.power_to_db(S, ref=np.max)

    # Espectrograma normalizado (media=0, std=1)
    S_norm = (S - np.mean(S)) / np.std(S)
    S_norm_db = librosa.power_to_db(S_norm - np.min(S_norm), ref=np.max)

    # Gráficos lado a lado
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='mel', ax=ax[0])
    ax[0].set_title("Sin normalizar")
    librosa.display.specshow(S_norm_db, sr=sr, x_axis='time', y_axis='mel', ax=ax[1])
    ax[1].set_title("Normalizado")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":

    analyze_energy_distribution('src/data/raw_train/normal_id_00_00000005.wav', cfg)
    parameter_sweep('src/data/raw_train/normal_id_00_00000005.wav')
    compare_normalization_effect('src/data/raw_train/normal_id_00_00000005.wav', cfg)