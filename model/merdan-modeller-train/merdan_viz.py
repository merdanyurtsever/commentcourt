import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
from datetime import datetime
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, roc_auc_score
from sklearn.preprocessing import label_binarize

def save_visualizations(y_true, y_pred, model_name, data_path, hyperparams, out_dir, history=None):
    """
    Generates and saves:
    1. Actual vs Predicted Scatter (Regression View)
    2. Residual Histogram (Error View)
    3. Confusion Matrix (Classification View: 1-5 Stars)
    4. ROC Curve (Binary View: Positive vs Negative)
    5. Classification Report (Precision, Recall, F1, Support)
    6. Training History (Loss/Accuracy) - Only if history is provided
    7. Summary Card
    """
    
    # Create Timestamped Directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(out_dir, f"{model_name}_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    
    # Style Config
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({'figure.figsize': (10, 6), 'font.size': 10})
    
    # --- HELPER: Convert 0-1 Score back to 1-5 Stars ---
    def to_stars(scores):
        # 0.0 -> 1, 1.0 -> 5
        stars = np.round(scores * 4 + 1).astype(int)
        return np.clip(stars, 1, 5)

    y_true_stars = to_stars(y_true)
    y_pred_stars = to_stars(y_pred)

    # 1. SCATTER PLOT (Regression)
    plt.figure()
    jitter = np.random.normal(0, 0.02, size=len(y_true))
    sns.scatterplot(x=y_true+jitter, y=y_pred, alpha=0.3, edgecolor=None, color='#2ecc71')
    plt.plot([0, 1], [0, 1], color='#e74c3c', lw=2, linestyle='--', label='Ideal')
    plt.title(f"Actual vs Predicted: {model_name}")
    plt.xlabel("True Score (0.0 - 1.0)")
    plt.ylabel("Predicted Score")
    plt.legend()
    plt.savefig(os.path.join(run_dir, 'plot_scatter.png'), dpi=150)
    plt.close()

    # 2. RESIDUALS
    plt.figure()
    sns.histplot(y_true - y_pred, bins=50, kde=True, color='#3498db')
    plt.axvline(x=0, color='red', linestyle='--')
    plt.title(f"Residual Error Distribution: {model_name}")
    plt.savefig(os.path.join(run_dir, 'plot_residuals.png'), dpi=150)
    plt.close()

    # 3. CONFUSION MATRIX (1-5 Stars)
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_true_stars, y_pred_stars, labels=[1,2,3,4,5])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=[1,2,3,4,5], yticklabels=[1,2,3,4,5])
    plt.title(f"Confusion Matrix (1-5 Stars)\n{model_name}")
    plt.xlabel("Predicted Star")
    plt.ylabel("True Star")
    plt.savefig(os.path.join(run_dir, 'plot_confusion_matrix.png'), dpi=150)
    plt.close()

    # 4. ROC CURVE (Binary: Pos >= 4, Neg <= 2, Ignore 3)
    # We filter out '3 star' (Neutral) to get a clean Positive/Negative curve
    mask = y_true_stars != 3
    if np.sum(mask) > 0:
        y_bin_true = (y_true_stars[mask] >= 4).astype(int)
        y_bin_pred = y_pred[mask] # Use continuous score for ROC
        
        fpr, tpr, _ = roc_curve(y_bin_true, y_bin_pred)
        roc_auc = auc(fpr, tpr)
        
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve (Binary: Pos vs Neg)\n{model_name}')
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(run_dir, 'plot_roc_curve.png'), dpi=150)
        plt.close()

    # 5. CLASSIFICATION REPORT (Text)
    report = classification_report(y_true_stars, y_pred_stars, labels=[1,2,3,4,5], zero_division=0)
    with open(os.path.join(run_dir, 'classification_report.txt'), 'w') as f:
        f.write(f"Model: {model_name}\n")
        f.write("-" * 30 + "\n")
        f.write(report)
        
    # 6. TRAINING HISTORY (Loss/Accuracy Curves)
    if history:
        # Loss
        plt.figure()
        plt.plot(history['train_loss'], label='Train Loss', color='blue')
        plt.plot(history['val_loss'], label='Val Loss', color='orange')
        plt.title(f"Training Loss Curve: {model_name}")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(run_dir, 'plot_loss_curve.png'), dpi=150)
        plt.close()
        
        # Accuracy (if available) - we usually track MSE for regression, but if you tracked custom acc:
        if 'val_acc' in history or 'val_mae' in history:
            metric_key = 'val_acc' if 'val_acc' in history else 'val_mae'
            train_key = 'train_acc' if 'train_acc' in history else 'train_mae'
            plt.figure()
            plt.plot(history.get(train_key, []), label=f'Train {metric_key}', color='green')
            plt.plot(history.get(metric_key, []), label=f'Val {metric_key}', color='red')
            plt.title(f"Training Metric Curve: {model_name}")
            plt.legend()
            plt.savefig(os.path.join(run_dir, 'plot_metric_curve.png'), dpi=150)
            plt.close()

    # 7. SUMMARY CARD
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.axis('off')
    text = f"SUMMARY: {model_name}\n"
    text += f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
    text += "-" * 30 + "\n"
    text += f"MSE:  {mse:.5f}\n"
    text += f"MAE:  {mae:.5f}\n"
    text += f"R2:   {r2:.5f}\n"
    text += f"AUC:  {roc_auc:.5f}" if 'roc_auc' in locals() else ""
    text += "\n\nCONFIG:\n"
    for k, v in hyperparams.items():
        text += f"{k}: {v}\n"
    
    ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=12, va='top', fontfamily='monospace')
    plt.savefig(os.path.join(run_dir, 'summary_card.png'), dpi=150)
    plt.close()
    
    print(f"\n[Visuals Generated] Saved to: {run_dir}")