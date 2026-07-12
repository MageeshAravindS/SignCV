import os
import json
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import itertools
from model import SignLanguageMLP

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion Matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=14, fontweight='bold', pad=15)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, fontsize=10)
    plt.yticks(tick_marks, classes, fontsize=10)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black",
                 fontsize=11, fontweight='bold')

    plt.ylabel('True label', fontsize=12, fontweight='semibold')
    plt.xlabel('Predicted label', fontsize=12, fontweight='semibold')
    plt.tight_layout()

def main():
    data_dir = 'data'
    csv_path = os.path.join(data_dir, 'sign_landmarks.csv')
    weights_path = os.path.join(data_dir, 'model.pth')
    mapping_path = os.path.join(data_dir, 'labels.json')
    
    # Check if necessary files exist
    if not (os.path.exists(csv_path) and os.path.exists(weights_path) and os.path.exists(mapping_path)):
        print("Error: Missing dataset, model weights, or class label map.")
        print("Please run python/train.py first to train a model.")
        return
        
    # Load class mapping
    with open(mapping_path, 'r') as f:
        mapping = json.load(f)
    # Sort classes by index
    classes = [mapping[str(i)] for i in range(len(mapping))]
    
    # Load and split dataset
    print(f"Loading dataset from {csv_path} for evaluation...")
    df = pd.read_csv(csv_path)
    X = df.iloc[:, 1:].values
    y_raw = df.iloc[:, 0].values
    
    label_encoder = LabelEncoder()
    # Ensure encoder fits the exact order in mapping
    label_encoder.classes_ = np.array(classes)
    y = label_encoder.transform(y_raw)
    
    _, X_val, _, y_val = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
    
    # Load model
    print(f"Loading model weights from {weights_path}...")
    model = SignLanguageMLP(num_classes=len(classes))
    model.load_state_dict(torch.load(weights_path))
    model.eval()
    
    # Run evaluation
    print("Running inference on validation set...")
    inputs = torch.tensor(X_val, dtype=torch.float32)
    with torch.no_grad():
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        
    y_pred = preds.numpy()
    
    # Generate reports
    print("\n" + "="*60)
    print("  CLASSIFICATION METRICS REPORT")
    print("="*60)
    report = classification_report(y_val, y_pred, target_names=classes)
    print(report)
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_val, y_pred)
    
    # Save text report to file
    report_txt_path = os.path.join(data_dir, 'evaluation_report.txt')
    with open(report_txt_path, 'w') as f:
        f.write("SignCV Model Evaluation Metrics\n")
        f.write("="*40 + "\n\n")
        f.write(report)
    print(f"Saved text classification report to {report_txt_path}")
    
    # Plot and save confusion matrix chart
    plt.figure(figsize=(8, 7))
    plot_confusion_matrix(cm, classes=classes, title='SignCV Gestures - Confusion Matrix')
    
    cm_path = os.path.join(data_dir, 'confusion_matrix.png')
    plt.savefig(cm_path, dpi=150)
    print(f"Saved Confusion Matrix visualization to {cm_path}")
    
if __name__ == '__main__':
    main()
