
```markdown
# Conformer-AMC: Automatic Modulation Classification 📡
作者：张治柯｜北京交通大学
Author: Zhike Zhang | Beijing Jiaotong University

A high-performance deep learning pipeline for Automatic Modulation Classification (AMC) using the RadioML 2018 dataset. This project implements a hybrid **ConformerAMC** architecture that bridges local feature extraction (CNN) with global sequence modeling (Transformer), optimized for extreme noise environments.

## ✨ Key Features

* **Hybrid Architecture (Conformer):** Integrates 1D ResNet blocks (for local waveform feature extraction and downsampling) with a Transformer Encoder (`d_model=256`) for long-range temporal dependency modeling.
* **Curriculum Learning (SNR-based):** Dynamically adjusts the Signal-to-Noise Ratio (SNR) threshold during training (e.g., `15dB` $\rightarrow$ `10dB` $\rightarrow$ `0dB` $\rightarrow$ `-20dB`), allowing the model to learn clean features before tackling extreme noise, effectively preventing gradient shock.
* **High-Throughput Training:** Fully utilizes multi-GPU acceleration (`nn.DataParallel`), Automatic Mixed Precision (AMP), and large batch sizes for industrial-grade training speed.
* **Advanced Optimization:** Built with a sophisticated learning rate scheduler (Linear Warmup + Cosine Annealing), Label Smoothing (`0.1`), and Gradient Clipping (`max_norm=1.0`) to ensure stable convergence on heavily corrupted datasets.
* **Production Ready (ONNX):** Includes an independent script to export the trained PyTorch model to ONNX format. Supports dynamic batch axes (`dynamic_axes`) and operator folding (`opset_version=14`), ready for TensorRT or C++ edge deployment.

## 📂 Project Structure

```text
RadioML/
├── configs/
│   └── train_config.yaml      # Hyperparameters and dataset paths
├── dataloaders/
│   └── amc_dataset.py         # Custom PyTorch Dataset for HDF5 parsing
├── data_splits/               # Pre-split train/val/test indices (e.g., train_snrs.npy)
├── models/
│   ├── cnn_transformer.py     # ConformerAMC model definition
│   └── loss.py                # Loss factory (CrossEntropy + Label Smoothing)
├── utils/
│   └── visualize.py           # Evaluation plotting (Confusion Matrix & Acc vs SNR)
├── train.py                   # Main training loop with curriculum learning
├── test.py                    # Independent inference and evaluation script
└── export_onnx.py             # Export model to ONNX format
```

## 🚀 Quick Start

### 1. Training
Configure your parameters in `configs/train_config.yaml`, then start the training pipeline. The script automatically handles SNR-based curriculum stages.
```bash
python3 train-one_shot.py
```
*Checkpoints will be automatically saved to the `checkpoints/` directory.*

### 2. Testing & Inference
Evaluate the trained model on the validation/test set to get the overall accuracy and generate prediction arrays.
```bash
python3 test.py
```

### 3. Visualization
Generate a detailed performance report, including a high-order normalized confusion matrix and an Accuracy vs. SNR curve.
```bash
cd utils
python3 visualize.py
```
*Plots will be saved to `values/visualizations/`.*

### 4. Export to ONNX
Export the trained model to a highly optimized ONNX graph for downstream deployment (e.g., TensorRT).
```bash
python3 export_onnx.py
```

## 🛠️ Environment & Dependencies

* Python 3.8+
* PyTorch 2.0+ (CUDA enabled)
* NumPy, h5py, PyYAML, tqdm
* Matplotlib, Seaborn, scikit-learn (for visualization)

---

