
# ECG Arrhythmia Detection with Self-Supervised Learning

A deep learning project exploring ECG arrhythmia classification using Transformer-based architectures and self-supervised representation learning.

## Overview

This project aims to develop an ECG classification pipeline using 12-lead electrocardiogram (ECG) signals from the PTB-XL dataset.

The goal is to learn meaningful ECG representations through contrastive self-supervised pretraining and evaluate their usefulness for arrhythmia classification, particularly Atrial Fibrillation (AFib) detection.

## Dataset

- Dataset: PTB-XL
- Source: PhysioNet
- Signal: 12-lead ECG
- Sampling frequency: 100 Hz
- Recording duration: 10 seconds
- Current development subset: 1,120 recordings
- Classes: Normal Sinus Rhythm (SR) and Atrial Fibrillation (AFib)

## Model Architecture

The current implementation includes:

- 1D convolutional patch embedding
- Positional embeddings
- Multi-head self-attention
- Transformer encoder blocks
- Classification head
- Attention weight extraction

## Technologies

- Python
- PyTorch
- NumPy
- Pandas
- WFDB
- Scikit-learn
- Matplotlib

## Project Structure

```text
ecg-arrhythmia-detection-ssl/
├── data/
│   ├── raw/
│   └── processed/
├── models/
│   └── transformer.py
├── src/
│   ├── download_selected.py
│   ├── prepare_labels.py
│   ├── prepare_dataset.py
│   ├── explore_ecg.py
│   └── test_model.py
├── results/
├── requirements.txt
└── README.md
```

## Current Progress

- [x] PTB-XL metadata download
- [x] ECG signal download for 1,120 selected recordings
- [x] Normal and AFib label preparation
- [x] ECG Transformer implementation
- [x] Multi-head attention extraction
- [x] Transformer forward-pass testing
- [ ] Dataset preprocessing and validation
- [ ] Self-supervised contrastive pretraining
- [ ] Fine-tuning for AFib classification
- [ ] Supervised baseline comparison
- [ ] Model evaluation
- [ ] Attention-based explainability

## Running the Project

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Test the Transformer:

```bash
python src/test_model.py
```

## Disclaimer

This is an ongoing research and learning project. It is not intended for clinical diagnosis or medical decision-making.
