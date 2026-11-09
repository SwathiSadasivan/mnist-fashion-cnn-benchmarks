import os
import matplotlib
matplotlib.use('Agg')  # Must be set before importing pyplot for cloud/headless runs

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def save_confusion_matrix(cm: np.ndarray, classes: list, output_path: str = "metrics/confusion_matrices/cm.png"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def save_loss_curves(train_losses: list, val_losses: list, output_path: str = "metrics/loss_curves/loss.png"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.figure(figsize=(8, 5))
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, label="Train Loss", marker="o")
    plt.plot(epochs, val_losses, label="Validation Loss", marker="o")
    plt.title("Training & Validation Loss Curves")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()