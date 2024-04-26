import torch
from torch import nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader

class BidirectionalLSTM(pl.LightningModule):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout_rate=0.5):
        super(BidirectionalLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=input_dim,
                            hidden_size=hidden_dim,
                            num_layers=num_layers,
                            dropout=dropout_rate,
                            bidirectional=True,
                            batch_first=True)
        
        self.out_layer = nn.Linear(hidden_dim * 2, output_dim)
        self.cosine_loss = nn.CosineEmbeddingLoss()

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_time_step = lstm_out[:, -1, :]
        output = self.out_layer(last_time_step)
        return output

    def custom_loss(self, y_hat, y):
        # Assuming y_hat and y are ordered as [x, y, z, qw, qx, qy, qz]
        position_hat = y_hat[:, :3]
        quaternion_hat = y_hat[:, 3:]
        position_true = y[:, :3]
        quaternion_true = y[:, 3:]

        # MSE loss for position
        position_loss = nn.functional.mse_loss(position_hat, position_true)
        
        # Normalize quaternion predictions and true quaternions
        quaternion_hat = torch.nn.functional.normalize(quaternion_hat, p=2, dim=1)
        quaternion_true = torch.nn.functional.normalize(quaternion_true, p=2, dim=1)
        
        # Cosine loss for quaternion (using a label of 1 to indicate maximum similarity)
        quaternion_loss = self.cosine_loss(quaternion_hat, quaternion_true, torch.ones(quaternion_hat.size(0)).to(quaternion_hat.device))

        # Combine losses
        total_loss = position_loss + quaternion_loss
        return total_loss

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.custom_loss(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.custom_loss(y_hat, y)
        self.log('val_loss', loss)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.custom_loss(y_hat, y)
        self.log('test_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer
