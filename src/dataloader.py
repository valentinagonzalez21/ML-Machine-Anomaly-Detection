import torch
from torch.utils.data import Dataset
import numpy as np
import os
from src.config_loader import load_config

cfg = load_config()
PROC_DIR = cfg["paths"]["processed_data_dir"]
#PROC_DIR = cfg["paths"]["processed_data_test_dir"]

class MelSpectrogramDataset(Dataset):
    def __init__(self, data_dir=PROC_DIR):
        self.files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".npy")]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        mel = np.load(self.files[idx])
        mel = torch.tensor(mel, dtype=torch.float32)

      #  if cfg.get("audio", {}).get("normalize", False):
        #    mel = (mel - mel.mean()) / (mel.std() + 1e-6)

        return mel
