import torch
import torch.nn as nn

class FrameWiseBiLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_classes, dropout=0.1):
        super(FrameWiseBiLSTM, self).__init__()

        self.input_norm = nn.LayerNorm(input_dim)
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_classes)
        )
    def forward(self, x):
        x = self.input_norm(x)
        lstm_out, _ = self.lstm(x)
        logits = self.classifier(lstm_out)
        return logits