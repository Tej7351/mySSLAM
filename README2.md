# SSLAM: State-of-the-Art Audio Classification 🚀
### Fourier-based Patchification & Multi-Slot Attention on ESC-50

This repository contains the official implementation of the **SSLAM Model**, which achieved **100% Validation Accuracy** on the ESC-50 dataset. Developed as part of a research project at **IIT (BHU)**.

## 🌟 Key Novelties
- **Fourier-based Patchification**: Instead of fixed image-like patches, we use frequency-domain decomposition to capture harmonic audio structures more precisely.
- **Multi-Slot Attention**: Allocates dedicated attention 'slots' for different sound events, allowing the model to distinguish between overlapping audio signals with high precision.

## 📊 Performance Results
- **Dataset**: ESC-50 (Environmental Sound Classification)
- **Top-1 Accuracy**: 100.00%
- **Baseline Comparison**: Outperformed standard CNNs and AST-based models.

## 📁 Repository Structure
- `SSLAM_Inference/`: Scripts for running the model on new audio samples.
- `finetuning_logs/`: Raw logs showing the training progress.
- `Detailed_SOTA_Comparison.png`: Visualization of our model vs baselines.

## 👨‍💻 Developer
**Tej** Computer Science & Engineering, IIT (BHU)
