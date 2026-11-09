# PyTorch & Azure ML Deep Learning Benchmarks: Fashion-MNIST & MNIST

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2-red.svg)](https://pytorch.org/)
[![Azure ML](https://img.shields.io/badge/Azure%20ML-v2%20CLI-0078D4.svg)](https://azure.microsoft.com/en-us/products/machine-learning/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end, modular machine learning engineering project benchmarking classical baselines (Logistic Regression GLM, LightGBM) against custom PyTorch Convolutional Neural Networks (CNNs) on **Fashion-MNIST** and **MNIST** datasets.

This repository demonstrates production-grade software architecture, automated experiment tracking via MLflow, unit testing with `pytest`, editable package packaging, and cloud-scale model training using **Azure Machine Learning Studio (v2 CLI)**.

---

## 📌 Executive Summary & Key Results

| Model Architecture | Framework / Algorithm | Dataset | Input Dim / Parameters | Validation Accuracy | Training Target |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **GLM Baseline** | Scikit-Learn `LogisticRegression` | Fashion-MNIST | $784$ (Flattened) | ~78.40% | Local CPU |
| **GBM Baseline** | LightGBM `LGBMClassifier` | Fashion-MNIST | 100 Trees | ~84.50% | Local CPU |
| **Baseline FNN** | PyTorch 2-Layer MLP | Fashion-MNIST | ~101,770 params | ~86.20% | Local CPU |
| **Custom 2D CNN** | PyTorch Conv2D + BatchNorm | Fashion-MNIST | ~1.60M params | **90.60%** | Local / Azure ML |
| **Custom 2D CNN** | PyTorch Conv2D + BatchNorm | MNIST | ~1.60M params | **99.10%** | Local / Azure ML |

---

## 🏗️ Repository Architecture

The project follows standard Python packaging guidelines with modular source code under `src/` and unit tests in `tests/`:

```text
mnist-fashion-cnn-benchmarks/
├── .venv/                      # Python virtual environment
├── data/                       # Downloaded datasets (raw & processed)
│   └── raw/
├── metrics/                    # Generated metrics & visual artifacts
│   ├── confusion_matrices/     # Confusion matrix heatmaps
│   └── loss_curves/            # Training/Validation loss plots
├── src/                        # Modular source code package
│   ├── __init__.py
│   ├── dataset.py              # Torchvision dataloaders & preprocessing
│   ├── evaluate.py             # Plotting utilities & metric evaluation
│   ├── train.py                # Core training loop with MLflow logging
│   └── models/
│       ├── __init__.py         # Model package exports
│       ├── baselines.py        # FNN, GLM, and LightGBM implementations
│       └── cnn.py              # Custom 2D CNN PyTorch architecture
├── tests/                      # Unit testing suite
│   ├── __init__.py
│   └── test_models.py          # Pytest validation for model forward passes
├── azure_job.yml               # Azure ML v2 CLI submission manifest
├── pyproject.toml              # Editable package configuration
├── requirements.txt            # Environment dependencies
└── README.md                   # Project documentation
```

---

## 🧠 Model Architecture (`CustomCNN`)

The `CustomCNN` architecture (`src/models/cnn.py`) utilizes 2D spatial convolutions combined with Batch Normalization and Dropout regularization:

1. **Convolutional Block 1:**
   * Input Channels: $1$ (Grayscale $28 \times 28$)
   * Output Channels: $32$ ($3 \times 3$ kernel, stride 1, padding 1)
   * 2D Batch Normalization + ReLU Activation
2. **Convolutional Block 2:**
   * Input Channels: $32$
   * Output Channels: $64$ ($3 \times 3$ kernel, stride 1, padding 1)
   * 2D Batch Normalization + ReLU Activation
3. **Pooling Layer:**
   * Max Pooling ($2 \times 2$ window, stride 2) $\rightarrow$ Output dimension: $64 \times 14 \times 14$
4. **Fully Connected Classifier:**
   * Dropout ($p=0.25$)
   * FC Layer 1: $12,544 \rightarrow 128$ units + ReLU
   * FC Layer 2 (Output): $128 \rightarrow 10$ class logits

---

## 🛠️ Local Environment & Installation

### 1. Clone & Setup Virtual Environment

```bash
git clone https://github.com/SwathiSadasivan/mnist-fashion-cnn-benchmarks.git
cd mnist-fashion-cnn-benchmarks

python3 -m venv .venv
source .venv/bin/activate
```

### 2. Editable Package Installation

Installing the project as an editable package (`pip install -e .`) links the project root to `.venv` site-packages, resolving import paths cleanly across `src/`:

```bash
pip install -e .
pip install torch torchvision scikit-learn lightgbm matplotlib seaborn mlflow pytest azure-ai-ml
```

> **macOS Note:** LightGBM requires OpenMP runtime dependencies on macOS ARM processors. If missing, install via Homebrew:
> ```bash
> brew install libomp
> ```

---

## 🧪 Unit Testing & Local Execution

### Run Unit Tests
Execute the unit test suite to verify model output tensor dimensions and batch processing:

```bash
pytest tests/
```

### Run Local Training
Execute a 2-epoch benchmark run locally using the Adam optimizer:

```bash
python3 src/train.py --dataset fashion_mnist --optimizer adam --lr 0.001 --epochs 2
```

Artifacts (confusion matrix and loss curves) are auto-saved to `metrics/`.

---

## ☁️ Azure Machine Learning Cloud Deployment

This project supports distributed cloud training runs on **Azure Machine Learning Studio (v2 CLI)**.

### 1. Authenticate Azure CLI & Set Workspace Defaults

```bash
az login
az extension add -n ml
az configure --defaults group=rg-mnist-benchmarks workspace=mlw-mnist-benchmarks
```

### 2. Azure Job Manifest (`azure_job.yml`)

```yaml
$schema: https://azuremlschemas.azureedge.net/latest/commandJob.schema.json
code: .
command: python3 src/train.py --dataset fashion_mnist --optimizer adam --lr 0.001 --epochs 10
environment: azureml:AzureML-acpt-pytorch-2.2-cuda12.1@latest
compute: azure-cpu-cluster
experiment_name: fashion-mnist-cnn-benchmarks
display_name: PyTorch-Adam-FashionMNIST-CloudRun
```

### 3. Submit Cloud Training Job

```bash
az ml job create --file azure_job.yml
```

Monitor live metrics, logs (`user_logs/std_log.txt`), and MLflow tracking dashboards directly within **Azure Machine Learning Studio**.

---

## 📈 Experiment Tracking & Visualizations

Every run logs metrics (loss and accuracy per epoch) and automatically outputs graphical artifacts using MLflow:

* **Confusion Matrix:** `metrics/confusion_matrices/fashion_mnist_adam_cm.png`
* **Loss Curves:** `metrics/loss_curves/fashion_mnist_adam_loss.png`

---

## 📄 License
This project is licensed under the **MIT License**. See `LICENSE` for details.