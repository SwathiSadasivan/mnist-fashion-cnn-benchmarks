import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
import mlflow
from sklearn.metrics import accuracy_score, confusion_matrix
from src.dataset import get_data_loaders
from src.models.cnn import CustomCNN
from src.evaluate import save_confusion_matrix, save_loss_curves


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_preds, all_labels = [], []

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    return epoch_loss, epoch_acc


def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    val_loss = running_loss / len(loader.dataset)
    val_acc = accuracy_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)
    return val_loss, val_acc, cm


def main():
    parser = argparse.ArgumentParser(description="Train CNN on MNIST / Fashion-MNIST")
    parser.add_argument("--dataset", type=str, default="fashion_mnist", help="mnist or fashion_mnist")
    parser.add_argument("--optimizer", type=str, default="adam", choices=["adam", "sgd"])
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.25)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Using compute device: {device}")

    train_loader, test_loader = get_data_loaders(dataset_name=args.dataset, batch_size=args.batch_size)
    model = CustomCNN(dropout_rate=args.dropout).to(device)
    criterion = nn.CrossEntropyLoss()

    if args.optimizer.lower() == "adam":
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    else:
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)

    mlflow.start_run()
    mlflow.log_params(vars(args))

    train_losses, val_losses = [], []

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        v_loss, v_acc, _ = validate(model, test_loader, criterion, device)

        train_losses.append(tr_loss)
        val_losses.append(v_loss)

        mlflow.log_metric("train_loss", tr_loss, step=epoch)
        mlflow.log_metric("train_acc", tr_acc, step=epoch)
        mlflow.log_metric("val_loss", v_loss, step=epoch)
        mlflow.log_metric("val_acc", v_acc, step=epoch)

        print(f"Epoch [{epoch}/{args.epochs}] | Train Loss: {tr_loss:.4f} Acc: {tr_acc:.4f} | Val Loss: {v_loss:.4f} Acc: {v_acc:.4f}")

    _, final_acc, cm = validate(model, test_loader, criterion, device)
    print(f"\n✅ Final Validation Accuracy: {final_acc * 100:.2f}%")

    cm_path = f"metrics/confusion_matrices/{args.dataset}_{args.optimizer}_cm.png"
    loss_path = f"metrics/loss_curves/{args.dataset}_{args.optimizer}_loss.png"

    class_names = [str(i) for i in range(10)]
    save_confusion_matrix(cm, class_names, cm_path)
    save_loss_curves(train_losses, val_losses, loss_path)

    mlflow.log_artifact(cm_path)
    mlflow.log_artifact(loss_path)
    mlflow.end_run()


if __name__ == "__main__":
    main()