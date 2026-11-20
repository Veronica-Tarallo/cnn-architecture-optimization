# CNN Architecture Optimization & Error Analysis for Image Classification

An end-to-end Deep Learning pipeline implemented in Python (**TensorFlow / Keras**) to solve a binary image classification task on the `cycle_gan/horse2zebra` dataset (2,661 RGB images at $256 \times 256$ resolution).

Rather than relying on off-the-shelf pre-trained backbones, this project focuses on **custom architecture design from scratch**, mathematical analysis of training dynamics, hyperparameter optimization, and qualitative error inspection.

## Methodology & Modeling Progression

1. **Leakage-Free Stratified Pipeline:** Merged original splits and partitioned the 2,661 images into **Training (80%)**, **Validation (10%)**, and **Test (10%)** sets via stratified sampling prior to exploratory analysis to prevent data leakage. Applied in-graph pixel rescaling (`1./255`) and data augmentation.
2. **Architecture Engineering ("Deep & Light" Design):**
   * Diagnosed severe overfitting starting at epoch 4 in an initial high-capacity baseline CNN (~3.74M trainable parameters).
   * Systematically evaluated three alternative topologies, culminating in a 4-block convolutional architecture (**Model 4**) that reduced trainable parameters by **77%** (down to **~835k**), curbing memorization while improving feature abstraction.
3. **Training Dynamics & Hyperparameter Tuning:**
   * Applied **Exponential Moving Average (EMA)** smoothing to validation loss curves to separate stochastic mini-batch fluctuations from true overfitting onset, explicitly accounting for filter phase lag.
   * Executed a Grid Search across **Dropout** and **Learning Rate** paired with **Early Stopping** to select the optimal regularization regime.

## Evaluation & Qualitative Error Analysis

Evaluated on the unseen test set, the final model achieved:
* **Test Accuracy:** `94.01%`
* **Test Precision:** `97.14%`

A manual inspection of all **16 misclassified test samples** revealed two primary failure modes:
* **Texture Bias:** False positives triggered by periodic high-contrast background patterns (e.g., wooden fences or striped shadows behind horses).
* **High-Frequency Attenuation:** Loss of subtle stripe contrast under extreme lighting/overexposure after successive `MaxPooling` downsampling operations.

## Repository Structure

* `cnn_architecture_optimization.ipynb` - Complete Jupyter Notebook containing data preprocessing, exploratory analysis, model definitions, training logs, EMA diagnostics, Grid Search, and visual error analysis.

## How to Run

1. Open `cnn_architecture_optimization.ipynb` in **Google Colab** or a local **Jupyter** environment with `tensorflow`, `tensorflow-datasets`, `numpy`, `scikit-learn`, and `matplotlib` installed.
2. Run all cells sequentially to reproduce data loading, model training, and test set evaluation.
