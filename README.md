# Anomaly detection in machines
## Machine Learning Project

This project looks at industrial sound anomaly detection using dense and convolutional autoencoders trained on Mel-spectrograms.
It includes audio preprocessing, configuration handling, and PyTorch model definitions.

The repository contains:

- Audio preprocessing code (WAV → Mel-spectrogram → normalized .npy)
- A configurable training and testing pipeline
- Two autoencoder architectures:
    - Dense Autoencoder (fully-connected)
    - Convolutional Autoencoder (Conv2D + ConvTranspose2D)

## Files description

Project structure (relevant files/folders)

- `configs/`
    - `config.yaml` — Main configuration file used across preprocessing, training and testing. Contains dataset paths, audio and spectrogram parameters, model hyperparameters, and training options.

- `src/`
    - `config_loader.py` — Utilities to load and validate configuration values from `config.yaml` and environment overrides.
    - `dataloader.py` — Data loading utilities that wrap NumPy spectrogram `.npy` files and provide PyTorch `Dataset`/`DataLoader` compatible iterators. Handles batching and optional augmentation.
    - `preprocess_audio.py` — Audio preprocessing pipeline: read raw audio, compute Mel-spectrograms, normalize, and save processed `.npy` files. Parameters (sample rate, n_mels, frame length/stride) are taken from `config.yaml`.
    - `models/` — Model definitions and architecture implementations.
        - dense and convolutional autoencoders class definitions.
        - `trained/`: log of all performed runs, generated model, and test results.
    - `spectrogram_analysis/` — Helper scripts and visualization tools for inspecting spectrograms and example signals.
    - `training/` — Training scripts and routines. Entrypoints here orchestrate the model, optimizer, loss, checkpoint saving, and training loop; they use `config_loader` and `dataloader`.
    - `testing/` — Evaluation and testing routines. Compute reconstruction errors, anomaly scores, and optionally generate plots / ROC curves.
- `data/`
    - Contains the dataset separated by training and testing audios, as well as raw ones and already processed ones.

## How to use

Quick start

Prev: 
Download dataset.zip (https://umontevideo-my.sharepoint.com/:u:/g/personal/vgonzalez_correo_um_edu_uy/IQDsdHY29ysoSoEt6m4ZC4EEAUZ9Vzm7zaUjqDbA-Y4CWck?e=yIeB9w)
Decompress dataset.zip and place in data folder so that it has this structure:
- data
    - raw_test
    - raw_train

1. Install dependencies

2. Edit `configs/config.yaml` to set paths and hyperparameters for your environment.

3. Preprocess raw audio (example):

python src/preprocess_audio.py
Processed .npy will appear in a new processed_train folder.


4. Train a model (example):

Within train_autoencoder, set variable model_type = "dense" or "conv", to train a model.
python src.training.train_autoencoder.py --config configs/config.yaml


5. Evaluate / test a trained model (example):

Change path variables in preprocess_audio.py to point towards the test dataset.
Processed .npy will appear in a new processed_test folder. 
Change path in test_autoencoder.py to point toward the model you want to evaluate.
python src.testing.test_autoencoder.py

6. See results
In the models/trained folder, you will find the run to see the logged result metrics.


Notes

- The `configs/config.yaml` centralizes nearly all experiment settings; prefer changing it over editing scripts.
- Use the scripts under `spectrogram_analysis/` to inspect intermediate preprocessing results and to help tune spectrogram parameters.
- Model definitions live under `src/models/` — you can add new architectures following the existing autoencoder interfaces.
