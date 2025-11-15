import os
import numpy as np
import librosa
from config_loader import load_config 
import shutil

cfg = load_config() # load configuration

RAW_DIR = cfg["paths"]["raw_data_dir"]
PROC_DIR = cfg["paths"]["processed_data_dir"]

#RAW_DIR = cfg["paths"]["raw_data_test_dir"]
#PROC_DIR = cfg["paths"]["processed_data_test_dir"]

# Función que convierte un archivo .wav a espectograma de MEL (en dB).
def wav_to_mel(path):
    y, sr = librosa.load(path, sr=cfg["audio"]["sample_rate"])  # Carga el audio a un array 'y' y su sample rate 'sr'.
    S = librosa.feature.melspectrogram(
        y=y,                                        # Señal de audio
        sr=sr,                                      # Sample rate
        n_mels=cfg["audio"]["n_mels"],              # Nº de bandas Mel
        n_fft=cfg["audio"]["n_fft"],                # Tamaño de la ventana FFT
        hop_length=cfg["audio"]["hop_length"],      # Paso entre ventanas
        power=cfg["audio"]["power"]                 # Exponente de potencia
    )
    S_dB = librosa.power_to_db(S, ref=np.max)       # Paso a escala logarítmica (dB)
    return S_dB

#Normaliza el espectrograma a media 0 y std 1
def normalize_mel(S_dB):
    return (S_dB - np.mean(S_dB)) / (np.std(S_dB) + 1e-6)


 # Función que recorre todos los .wav y los convierte a .npy
def process_all():

    if os.path.exists(PROC_DIR):
        print(f"Borrando carpeta existente: {PROC_DIR}")
        shutil.rmtree(PROC_DIR)

    os.makedirs(PROC_DIR, exist_ok=True) # Crea la carpeta de salida si no existe
    for fname in os.listdir(RAW_DIR):
        if fname.endswith(".wav"):
            in_path = os.path.join(RAW_DIR, fname)
            mel = wav_to_mel(in_path)
            mel_norm = normalize_mel(mel)
            out_path = os.path.join(PROC_DIR, fname.replace(".wav", ".npy"))
            np.save(out_path, mel_norm)
            print(f"Procesado {fname} -> {out_path}")


if __name__ == "__main__":
    process_all()