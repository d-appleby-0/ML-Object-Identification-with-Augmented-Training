import matplotlib.pyplot as plt

def plot_training_metrics(history, save_path=None):
    epochs = history["epoch"]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 1. Loss Plot
    axes[0].plot(epochs, history["train_loss"], marker='o', color='crimson', label='Train Loss')
    axes[0].set_title("Training Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, linestyle='--', alpha=0.6)
    axes[0].legend()
    
    # 2. AUC & F1 Plot
    axes[1].plot(epochs, history["val_auc"], marker='s', color='navy', label='Val AUC')
    axes[1].plot(epochs, history["val_f1"], marker='^', color='teal', label='Val F1 (thr=0.5)')
    axes[1].set_title("Validation AUC & F1-Score")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].grid(True, linestyle='--', alpha=0.6)
    axes[1].legend()

    # 3. Sensitivity vs Specificity Plot
    axes[2].plot(epochs, history["val_sens"], marker='o', color='darkorange', label='Sensitivity (Recall)')
    axes[2].plot(epochs, history["val_spec"], marker='d', color='purple', label='Specificity')
    axes[2].set_title("Sensitivity vs Specificity (thr=0.5)")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Rate")
    axes[2].set_ylim(0.0, 1.0)
    axes[2].grid(True, linestyle='--', alpha=0.6)
    axes[2].legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Saved metric plots to {save_path}")
    plt.show()