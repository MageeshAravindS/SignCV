import torch
import torch.nn as nn

class SignLanguageMLP(nn.Module):
    def __init__(self, num_classes):
        super(SignLanguageMLP, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(63, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(64, 32),
            nn.ReLU(),
            
            nn.Linear(32, num_classes)
        )
        
    def forward(self, x):
        return self.fc(x)

class SignLanguageLSTM(nn.Module):
    def __init__(self, num_classes, hidden_size=64, num_layers=2):
        super(SignLanguageLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Bidirectional LSTM layer
        self.lstm = nn.LSTM(
            input_size=63,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.2 if num_layers > 1 else 0.0
        )
        
        # Classifier head
        # bidirectional doubles hidden_size output
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, num_classes)
        )
        
    def forward(self, x):
        # Input shape x: (batch_size, sequence_length, 63)
        # LSTM output shape: (batch_size, sequence_length, hidden_size * 2)
        out, _ = self.lstm(x)
        
        # Take the output from the final time step
        out = out[:, -1, :]
        
        # Classify
        return self.fc(out)

if __name__ == '__main__':
    # Test MLP model
    mlp = SignLanguageMLP(num_classes=10)
    test_in_mlp = torch.randn(4, 63)
    test_out_mlp = mlp(test_in_mlp)
    print("MLP Output Shape:", test_out_mlp.shape) # Should be (4, 10)
    
    # Test LSTM model
    lstm = SignLanguageLSTM(num_classes=5)
    test_in_lstm = torch.randn(4, 30, 63) # Batch=4, SeqLen=30 frames, Features=63
    test_out_lstm = lstm(test_in_lstm)
    print("LSTM Output Shape:", test_out_lstm.shape) # Should be (4, 5)
