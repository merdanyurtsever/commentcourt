import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
from datetime import datetime

def save_visualizations(y_true, y_pred, model_name, data_path, hyperparams, out_dir):
    """
    Generates and saves:
    1. Actual vs Predicted Scatter Plot
    2. Residual Histogram (Error Distribution)
    3. Metrics & Config Summary Image
    """
    
    # Create Timestamped Directory for this specific run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(out_dir, f"{model_name}_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    
    # Style Config
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({'figure.figsize': (10, 6), 'font.size': 12})
    
    # ---------------------------------------------------------
    # 1. Actual vs Predicted Scatter Plot
    # ---------------------------------------------------------
    plt.figure()
    # Add some jitter to x-axis so points don't overlap perfectly on vertical lines
    jittered_true = y_true + np.random.normal(0, 0.02, size=len(y_true))
    
    sns.scatterplot(x=jittered_true, y=y_pred, alpha=0.3, edgecolor=None, color='#2ecc71')
    
    # Ideal line
    plt.plot([0, 1], [0, 1], color='#e74c3c', lw=2, linestyle='--', label='Ideal Perfect Prediction')
    
    plt.title(f"Actual vs Predicted: {model_name}", fontsize=14, fontweight='bold')
    plt.xlabel("True Score (Normalized)")
    plt.ylabel("Predicted Score")
    plt.legend()
    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.05, 1.05)
    
    # Add Text Box with Data Source
    dataset_name = os.path.basename(data_path)
    plt.figtext(0.5, 0.01, f"Data: {dataset_name} | {timestamp}", ha="center", fontsize=10, color='gray')
    
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, 'plot_scatter.png'), dpi=150)
    plt.close()

    # ---------------------------------------------------------
    # 2. Residual Histogram (Error Distribution)
    # ---------------------------------------------------------
    residuals = y_true - y_pred
    
    plt.figure()
    sns.histplot(residuals, bins=50, kde=True, color='#3498db', edgecolor='white')
    
    plt.axvline(x=0, color='#e74c3c', linestyle='--', lw=2)
    plt.title(f"Error Distribution (Residuals): {model_name}", fontsize=14, fontweight='bold')
    plt.xlabel("Error (True - Predicted)")
    plt.ylabel("Count")
    
    # Annotate Mean Error
    mean_err = np.mean(residuals)
    std_err = np.std(residuals)
    plt.figtext(0.15, 0.8, f"Mean Error: {mean_err:.4f}\nStd Dev: {std_err:.4f}", 
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))
    
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, 'plot_residuals.png'), dpi=150)
    plt.close()

    # ---------------------------------------------------------
    # 3. Hyperparameters & Metrics Card
    # ---------------------------------------------------------
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    corr = np.corrcoef(y_true, y_pred)[0, 1]
    
    # Create a blank image for text
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.axis('off')
    
    text_content = f"RUN SUMMARY: {model_name}\n"
    text_content += f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
    text_content += "-" * 40 + "\n\n"
    
    text_content += "PERFORMANCE METRICS:\n"
    text_content += f"R2 Score:      {r2:.6f}\n"
    text_content += f"Pearson Corr:  {corr:.6f}\n"
    text_content += f"MSE:           {mse:.6f}\n"
    text_content += f"MAE:           {mae:.6f}\n\n"
    
    text_content += "-" * 40 + "\n"
    text_content += "CONFIGURATION:\n"
    text_content += f"Dataset: {dataset_name}\n\n"
    
    for key, val in hyperparams.items():
        text_content += f"{key}: {val}\n"
        
    ax.text(0.05, 0.95, text_content, transform=ax.transAxes, fontsize=12, 
            verticalalignment='top', fontfamily='monospace')
    
    plt.savefig(os.path.join(run_dir, 'summary_card.png'), dpi=150)
    plt.close()
    
    print(f"\n[Visuals Generated] Saved plots to: {run_dir}")