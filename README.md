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
  - `F1-score`, `Precision`, `Recall`, `ROC-AUC`
  - Live inference confidence tracking.

- **Live Inference**:
  - Real-time microphone capture via `sounddevice`.
  - Voice detection and classification loop in **Jupyter Notebook**.
  - Class and confidence printed on screen.

---

## 📁 Project Structure

```
FCV_project/
├── src/                      # Source modules
│   ├── data/                 # Dataset classes, augmentation, generators
│   ├── models/               # CNN architectures
│   ├── trainer/              # Training and evaluation pipeline
│   └── utils/                # Helper functions and model utils
├── notebooks/                # Experiments and live inference
├── best_model.pt             # Trained model checkpoint
├── requirements.txt          # Dependencies
├── config.yaml               # (optional) Training config
└── README.md                 # Project description
```
---

The detailed project report is available as a PDF:

[👉 Check the report (PDF)](reports/FoCV Report.pdf)

[👉 Check the report (PDF)]([reports/report.pdf](https://github.com/BartekKrzepkowski/FoCV_project/blob/main/report/FoCV%20Report.pdf))
