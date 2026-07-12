import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from model import SignLanguageMLP

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

class LandmarkDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        
    def __len__(self):
        return len(self.X)
        
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def create_mockup_dataset(file_path):
    print("Dataset CSV not found. Generating a mockup dataset for testing training...")
    classes = ['A', 'B', 'C', 'L', 'Y']
    samples_per_class = 120
    
    data = []
    for cls in classes:
        for _ in range(samples_per_class):
            # Create a mock 63-dimensional normalized coordinate array
            # We add some class-specific structure + gaussian noise
            base_vec = np.zeros(63)
            if cls == 'A': # Fist shape
                base_vec[::3] = np.random.normal(0.1, 0.05, 21)
                base_vec[1::3] = np.random.normal(0.2, 0.05, 21)
            elif cls == 'B': # Open hand
                base_vec[::3] = np.linspace(-0.5, 0.5, 21) + np.random.normal(0, 0.05, 21)
                base_vec[1::3] = np.linspace(-1.0, 0.0, 21) + np.random.normal(0, 0.05, 21)
            elif cls == 'C': # Curved
                base_vec[::3] = np.sin(np.linspace(0, np.pi, 21)) + np.random.normal(0, 0.05, 21)
            elif cls == 'L': # L-shape (thumb & index extended)
                base_vec[12] = 1.0 # thumb
                base_vec[24] = -1.0 # index tip y
            elif cls == 'Y': # Pinky & Thumb extended
                base_vec[12] = 1.0 # thumb
                base_vec[60] = 1.0 # pinky tip
                
            # Add small random noise
            noise = np.random.normal(0, 0.02, 63)
            data.append([cls] + list(base_vec + noise))
            
    # Save to CSV
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    cols = ['label'] + [f'coord_{i}' for i in range(63)]
    df = pd.DataFrame(data, columns=cols)
    df.to_csv(file_path, index=False)
    print(f"Mockup dataset successfully saved to {file_path}")

def main():
    data_dir = 'data'
    csv_path = os.path.join(data_dir, 'sign_landmarks.csv')
    
    # Generate mock data if file not found
    if not os.path.exists(csv_path):
        create_mockup_dataset(csv_path)
        
    # Read the dataset
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"Dataset Shape: {df.shape}")
    
    X = df.iloc[:, 1:].values # all coordinate columns
    y_raw = df.iloc[:, 0].values # labels
    
    # Encode string labels to integers
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)
    classes = label_encoder.classes_.tolist()
    num_classes = len(classes)
    
    # Save label mapping to JSON (needed for inference)
    mapping = {i: cls for i, cls in enumerate(classes)}
    mapping_path = os.path.join('data', 'labels.json')
    with open(mapping_path, 'w') as f:
        json.dump(mapping, f, indent=4)
    print(f"Saved class index mapping to {mapping_path}: {mapping}")
    
    # Train/Validation Split (80/20)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
    
    train_dataset = LandmarkDataset(X_train, y_train)
    val_dataset = LandmarkDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Instantiate Model
    model = SignLanguageMLP(num_classes=num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003, weight_decay=1e-4)
    
    # Training Loop
    epochs = 40
    train_losses = []
    val_losses = []
    val_accuracies = []
    
    print("\nStarting model training...")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            
        epoch_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_loss)
        
        # Validation Phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)
                
                _, predicted = torch.max(outputs, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
                
        epoch_val_loss = val_loss / len(val_loader.dataset)
        val_losses.append(epoch_val_loss)
        
        epoch_acc = correct / total
        val_accuracies.append(epoch_acc)
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:02d}/{epochs:02d} | Train Loss: {epoch_loss:.4f} | Val Loss: {epoch_val_loss:.4f} | Val Accuracy: {epoch_acc*100:.2f}%")
            
    # Save Model Weights
    weights_path = os.path.join('data', 'model.pth')
    torch.save(model.state_dict(), weights_path)
    print(f"\nTrained model weights saved to {weights_path}")
    
    # Plot curves
    plt.figure(figsize=(12, 5))
    
    # Loss Curve
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss', color='indigo', linewidth=2)
    plt.plot(val_losses, label='Val Loss', color='cyan', linewidth=2)
    plt.title('Training & Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # Accuracy Curve
    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies, label='Val Accuracy', color='emerald' if hasattr(plt.cm, 'emerald') else 'green', linewidth=2)
    plt.title('Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    
    plots_path = os.path.join('data', 'training_curves.png')
    plt.tight_layout()
    plt.savefig(plots_path)
    print(f"Training performance curves saved to {plots_path}")
    
if __name__ == '__main__':
    main()
