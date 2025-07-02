# 🎙️ FoCV Project – Speaker Verification using CNNs and Spectrograms

This project focuses on voice-based **binary speaker verification** using **Convolutional Neural Networks (CNNs)** trained on **log-Mel spectrograms** extracted from audio recordings.

The goal is to distinguish between **"allowed"** and **"non-allowed"** speakers, with the ability to **incrementally add new speakers** to the allowed class without retraining the entire model from scratch.

---

## 🔧 Key Features

- **Audio Preprocessing Pipeline**:
  - Normalize waveform to **−20 dB RMS** for consistent loudness.
  - Apply **Voice Activity Detection (VAD)** to remove silence.
  - **Resample** audio to 16kHz for uniform time grid.
  - Apply **pre-emphasis filter** (0.97) to boost high frequencies.
  - Generate **log-Mel spectrograms**.

- **Custom CNN Architecture**:
  - Configurable blocks (channels, batch norm, activation, skip connections).
  - Fully trainable from scratch on spectrogram inputs.
  - Modular training pipeline with hyperparameter grid search.

- **Dataset Preparation**:
  - Stratified split across genders and labels.
  - Balanced sampling of speaker identities.
  - Replay buffer for controlled fine-tuning on new speakers.

- **Evaluation Metrics**:
  - `F1-score`, `Precision`, `Recall`
  - Live inference confidence tracking.

- **Live Inference**:
  - Real-time microphone capture via `sounddevice`.
  - Voice detection and classification loop in **Jupyter Notebook**.
  - Class and confidence printed on screen.

---

## 📁 Project Structure

- **root/**
    - **models/** – models checkpoints
    - **notebooks/** – Jupyter notebooks for experiments and analysis
    - **report/** – detailed report of the project
    - **src/** – project source code:
        - **configs/** – configuration files
        - **data/** – modules for data loading and processing
        - **modules/** – main project modules (e.g., network layers, ML components)
        - **trainer/** – model training logic
        - **utils/** – utility/helper functions
        - **visualization/** – scripts for data and results visualization
    - **.gitignore** – specifies files/folders to ignore in the repository
    - **main_new_speaker_training.py** – script for new speaker training
    - **main.py** – main script for running the project
    - **README.md** – project documentation
    - **run_main_new_speaker.sh** – bash script for running new speaker training
    - **run_main.sh** – bash script for running the main process

---

The detailed project report is available as a PDF:

[👉 Check the report (PDF)](report/FoCV%20Report.pdf)
