import argparse
from pathlib import Path

import torch
from torch import nn, optim
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, confusion_matrix, classification_report

import matplotlib.pyplot as plt

from data_split import split_dataset
from data_loaders import get_loaders
from model import build_model
import torch.nn.functional as F
import time

plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 15,        # Default font size
    "axes.labelsize": 15,   # Axis label font size
    "legend.fontsize": 15,  # Legend font size
    "mathtext.fontset": "stix",  # Math font set to STIX, which is close to Times New Roman
})

def val_evaluate(model, loader, device, plot_path=None):
    """Return {'acc': .., 'auc': ..}. Optionally save ROC curve plot."""
    model.eval()
    all_probs, all_labels = [], []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            probs = torch.softmax(model(x), dim=1)[:, 1].cpu().numpy()  # Positive class prob
            all_probs.extend(probs)
            all_labels.extend(y.numpy())

    acc = accuracy_score(all_labels, [p > 0.5 for p in all_probs])
    auc = roc_auc_score(all_labels, all_probs)

    if plot_path:
        fpr, tpr, _ = roc_curve(all_labels, all_probs)
        plt.figure()
        plt.plot(fpr, tpr, label=f"AUC={auc:.3f}")
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()

    return {"acc": acc, "auc": auc}


def test_evaluate(model, loader, device, plot_path=None):
    """Return {'acc': .., 'auc': .., 'cm': .., 'cr': ..}. Optionally save ROC curve plot."""
    model.eval()
    all_probs, all_labels, all_preds = [], [], []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            probs = torch.softmax(model(x), dim=1).cpu().numpy()  # Positive class prob
            preds = np.argmax(probs, axis=1)
            all_probs.extend(probs[:, 1])  # Use probability of positive class
            all_labels.extend(y.cpu().numpy())
            all_preds.extend(preds)

    # Calculate metrics
    acc = accuracy_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_probs)
    cm = confusion_matrix(all_labels, all_preds)
    cr = classification_report(all_labels, all_preds, target_names=['Class 0', 'Class 1'])

    # Print confusion matrix and classification report
    print("Confusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(cr)

    # Optionally plot ROC curve
    if plot_path:
        fpr, tpr, _ = roc_curve(all_labels, all_probs)
        plt.figure()
        plt.plot(fpr, tpr, label=f"AUC={auc:.3f}")
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()

    return {"acc": acc, "auc": auc, "cm": cm, "cr": cr}


def train(args):
    # 1) Split once (idempotent)
    split_dir = Path(args.workdir) / "splits"
    if not split_dir.exists():
        print("[Stage 1] Creating dataset splits ...")
        split_dataset(args.src_root, split_dir, ratio=(0.8, 0.1, 0.1), seed=args.seed)
    else:
        print("[Stage 1] Splits found – skipping.")

    # 2) Data
    print("[Stage 2] Building dataloaders ...")
    train_loader, val_loader, test_loader = get_loaders(split_dir, batch_size=args.batch_size)

    # 3) Model
    print("[Stage 3] Preparing model ...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(arch=args.arch).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Add Cosine Annealing Learning Rate Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_auc = 0.0
    out_dir = Path(args.workdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # For recording metrics during training
    train_losses = []
    val_accs = []
    val_aucs = []
    lrs = []

    # 4) Training loop
    print("[Stage 4] Training ...")
    StartTime = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            # Convert labels to one-hot encoding
            one_hot_y = F.one_hot(y, num_classes=2).float()
            optimizer.zero_grad()
            pred_x = model(x)
            loss = criterion(pred_x, one_hot_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        # Update learning rate
        scheduler.step()

        # Record training loss
        epoch_loss /= len(train_loader)
        train_losses.append(epoch_loss)

        # Record learning rate
        current_lr = optimizer.param_groups[0]['lr']
        lrs.append(current_lr)

        # Validation
        val_metrics = val_evaluate(model, val_loader, device)
        val_accs.append(val_metrics['acc'])
        val_aucs.append(val_metrics['auc'])

        print(f"Epoch {epoch:02d} | train_loss={epoch_loss:.4f} | val_acc={val_metrics['acc']:.4f} | val_auc={val_metrics['auc']:.4f} | time={time.time()-StartTime:.2f}s")
        if val_metrics["auc"] > best_auc:
            best_auc = val_metrics["auc"]
            torch.save(model.state_dict(), out_dir / "best_model.pt")


        # Print and write log
        line = (f"Epoch {epoch:02d} | "
                    f"train_loss={epoch_loss:.4f} | "
                    f"val_acc={val_metrics['acc']:.4f} | "
                    f"val_auc={val_metrics['auc']:.4f} | "
                    f"lr={current_lr:.6f} | "
                    f"time={time.time()-StartTime:.2f}s\n")
        with open(out_dir / 'train_log.txt', 'a') as f_log:
            f_log.write(line)

    # 5) Plot metrics during training
    plot_training_metrics(train_losses, val_accs, val_aucs, lrs, out_dir / "training_metrics.png")

    # 6) Test
    print("[Stage 5] Testing best model ...")
    model.load_state_dict(torch.load(out_dir / "best_model.pt", map_location=device))
    test_metrics = test_evaluate(model, test_loader, device, plot_path=out_dir / "roc_curve.png")
    print(f"Test acc={test_metrics['acc']:.4f} | Test AUC={test_metrics['auc']:.4f}")
    print(f"ROC curve saved to {out_dir / 'roc_curve.png'}")


def plot_training_metrics(train_losses, val_accs, val_aucs, lrs, save_path):
    """Plot training loss, validation accuracy, validation AUC, and learning rate changes"""
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))

    # Plot training loss
    axs[0, 0].plot(train_losses, label='Training Loss')
    axs[0, 0].set_title('Training Loss')
    axs[0, 0].set_xlabel('Epoch')
    axs[0, 0].set_ylabel('Loss')
    axs[0, 0].legend()

    # Plot validation accuracy
    axs[0, 1].plot(val_accs, label='Validation Accuracy')
    axs[0, 1].set_title('Validation Accuracy')
    axs[0, 1].set_xlabel('Epoch')
    axs[0, 1].set_ylabel('Accuracy')
    axs[0, 1].legend()

    # Plot validation AUC
    axs[1, 0].plot(val_aucs, label='Validation AUC')
    axs[1, 0].set_title('Validation AUC')
    axs[1, 0].set_xlabel('Epoch')
    axs[1, 0].set_ylabel('AUC')
    axs[1, 0].legend()

    # Plot learning rate
    axs[1, 1].plot(lrs, label='Learning Rate')
    axs[1, 1].set_title('Learning Rate')
    axs[1, 1].set_xlabel('Epoch')
    axs[1, 1].set_ylabel('Learning Rate')
    axs[1, 1].legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=600)
    plt.close()
    print(f"Training metrics plot saved to {save_path}")


def parse_args():
    p = argparse.ArgumentParser(description="Train binary lens classifier")
    p.add_argument("--src_root", type=str, default="dataset", help="Original dataset root")
    p.add_argument("--workdir", type=str, default="workdir", help="Outputs & split data dir")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--arch", type=str, default="wtresnet50",
                  help="Model architecture: davit_tiny, davit_small, resnet18, resnet50, wtresnet50 ...")

    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())