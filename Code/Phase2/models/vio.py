import torch
from torch import nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from models.helpers import BasicConvEncoder
from models.helpers import POLAUpdate
from models.helpers import DropPath
from models.helpers import to_2tuple
from models.losses import convert_Avec_to_A, A_vec_to_quat
from models.losses import quat_consistency_loss, quat_chordal_squared_loss, quaternion_to_matrix
from models.losses import GeodesicLoss

class VisualInertialOdometry(pl.LightningModule):
    def __init__(self, input_dim_inertial=6, hidden_dim_inertial=256, num_layers=2, memory_size=100, dropout=0.5, learning_rate=1e-3):
        super().__init__()
        self.fnet = nn.Sequential(
            BasicConvEncoder(output_dim=256, norm_fn='instance', dropout=dropout),
            POLAUpdate(embed_dim=256, depth=6, num_head=8, window_size=7, neig_win_num=1)
        )
        self.hidden_dim_inertial = hidden_dim_inertial 
        self.conv_reduce = nn.Conv2d(256, 128, kernel_size=1)
        
        self.adaptive_max_pool = nn.AdaptiveMaxPool2d((22, 28))
        self.relu = nn.ReLU()
        
        
        # Updated MultiheadAttention
        self.cross_attention = nn.MultiheadAttention(embed_dim=256, num_heads=8, dropout=dropout)
        
        self.conv_attn_map = nn.Conv2d(256, 32, kernel_size=1)

        # LSTM for processing inertial information
        self.lstm_inertial = nn.LSTM(input_size=input_dim_inertial,
                            hidden_size=hidden_dim_inertial,
                            num_layers=num_layers,
                            dropout=dropout,
                            bidirectional=True,
                            batch_first=True)

        
        self.fc1_position = nn.Linear(19840, 256)
        self.fc2_position = nn.Linear(256, 3)
        self.fc1_orientation = nn.Linear(19840, 256)
        self.fc2_orientation = nn.Linear(256, 4)
        
        self.geodesic = GeodesicLoss()
        self.mse = nn.L1Loss()

    def forward(self, image1, image2, imu):
        image1 = 2 * (image1 / 255.0) - 1.0
        image2 = 2 * (image2 / 255.0) - 1.0
        
        fmap1, fmap2 = self.fnet([image1, image2])
        # print(fmap1.shape, fmap2.shape)
        
        # Reduce dimensionality
        fmap1 = self.conv_reduce(self.adaptive_max_pool(fmap1))
        fmap2 = self.conv_reduce(self.adaptive_max_pool(fmap2))

        # Compute sequence length from spatial dimensions for concatenation
        sequence_length = fmap1.shape[2] * fmap1.shape[3]  # width * height
        fmap1 = fmap1.view(fmap1.shape[0], fmap1.shape[1], sequence_length)
        fmap2 = fmap2.view(fmap2.shape[0], fmap2.shape[1], sequence_length)

        # Concatenate feature maps along the channel dimension
        combined_fmaps = torch.cat([fmap1, fmap2], dim=1)  # Concatenate on channel dimension
        combined_fmaps = combined_fmaps.permute(2, 0, 1)  # Reshape for sequence model: (sequence_length, batch_size, embedding_dim)
        
        # Process inertial information with LSTM
        _, (hidden_inertial, _) = self.lstm_inertial(imu)
        # print(hidden_inertial.shape)

        concatenated_features = torch.cat([combined_fmaps, hidden_inertial], dim=0)
        # Cross-attention
        attn_output, _ = self.cross_attention(concatenated_features, concatenated_features, concatenated_features)
        
        attn_output = attn_output.permute(1, 2, 0).reshape(-1, 256, 20, 31)
        attn_output = self.conv_attn_map(attn_output)

        # Flatten the output from attention
        attn_output = attn_output.contiguous().view(attn_output.size(0), -1)
        
        # Position pathway
        pos = self.relu(self.fc1_position(attn_output))
        pos = self.fc2_position(pos)

        # Orientation pathway
        orient = self.relu(self.fc1_orientation(attn_output))
        orient = self.fc2_orientation(orient)
        # orient = A_vec_to_quat(orient).reshape(-1, 4)
        
        # print(pos.shape, orient.shape)
        # Combine position and orientation into a single output tensor
        res = torch.cat([pos, orient], dim=1)
    
        return res
    
    def freeze_fnet(self):
        """Freeze the fnet layer to prevent updates during training."""
        for param in self.fnet.parameters():
            param.requires_grad = False
    
    def custom_loss(self, y_hat, y):
        pos_hat, orient_hat = y_hat[:, :3], y_hat[:, 3:]
        pos, orient = y[:, :3], y[:, 3:]
        
        # print(y.shape, y_hat.shape)
        orient_hat = torch.nn.functional.normalize(orient_hat, p=2, dim=1)
        orient = torch.nn.functional.normalize(orient, p=2, dim=1)
        
        # print(pos_hat.shape, pos.shape, orient_hat.shape, orient.shape) #
        
        pos_loss = self.mse(pos_hat, pos)
        orient_loss = self.geodesic(quaternion_to_matrix(orient_hat), quaternion_to_matrix(orient))
        # orient_loss = quat_chordal_squared_loss(orient_hat, orient)
        
        return pos_loss + orient_loss
        
    def training_step(self, batch, batch_idx):
        x, imu, y = batch
        x1, x2 = x[:, 0], x[:, 1]
        y_hat = self(x1, x2, imu)
        loss = self.custom_loss(y_hat, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, imu, y = batch
        x1, x2 = x[:, 0], x[:, 1]
        y_hat = self(x1, x2, imu)
        loss = self.custom_loss(y_hat, y)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, imu, y = batch
        x1, x2 = x[:, 0], x[:, 1]
        y_hat = self(x1, x2, imu)
        loss = self.criterion(y_hat, y)
        self.log('test_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer
    
    
    

class VisualInertialOdometry2(pl.LightningModule):
    def __init__(self, input_dim_inertial=6, hidden_dim_inertial=256, num_layers=2, memory_size=100, dropout=0.5, learning_rate=1e-3):
        super().__init__()
        self.fnet = nn.Sequential(
            BasicConvEncoder(output_dim=256, norm_fn='instance', dropout=dropout),
            POLAUpdate(embed_dim=256, depth=6, num_head=8, window_size=7, neig_win_num=1)
        )
        self.hidden_dim_inertial = hidden_dim_inertial 
        self.conv_reduce = nn.Conv2d(256, 128, kernel_size=1)
        
        self.adaptive_max_pool = nn.AdaptiveMaxPool2d((22, 28))
        self.relu = nn.ReLU()
        
        
        # Updated MultiheadAttention
        self.cross_attention = nn.MultiheadAttention(embed_dim=256, num_heads=8, dropout=dropout)
        
        self.conv_attn_map = nn.Conv2d(256, 32, kernel_size=1)

        # LSTM for processing inertial information
        self.lstm_inertial = nn.LSTM(input_size=input_dim_inertial,
                            hidden_size=hidden_dim_inertial,
                            num_layers=num_layers,
                            dropout=dropout,
                            bidirectional=True,
                            batch_first=True)

        
        self.fc1_position = nn.Linear(19840, 256)
        self.fc2_position = nn.Linear(256, 3)
        self.fc1_orientation = nn.Linear(19840, 256)
        self.fc2_orientation = nn.Linear(256, 10)
        
        self.geodesic = GeodesicLoss()
        self.mse = nn.L1Loss()

    def forward(self, image1, image2, imu):
        image1 = 2 * (image1 / 255.0) - 1.0
        image2 = 2 * (image2 / 255.0) - 1.0
        
        fmap1, fmap2 = self.fnet([image1, image2])
        # print(fmap1.shape, fmap2.shape)
        
        # Reduce dimensionality
        fmap1 = self.conv_reduce(self.adaptive_max_pool(fmap1))
        fmap2 = self.conv_reduce(self.adaptive_max_pool(fmap2))

        # Compute sequence length from spatial dimensions for concatenation
        sequence_length = fmap1.shape[2] * fmap1.shape[3]  # width * height
        fmap1 = fmap1.view(fmap1.shape[0], fmap1.shape[1], sequence_length)
        fmap2 = fmap2.view(fmap2.shape[0], fmap2.shape[1], sequence_length)

        # Concatenate feature maps along the channel dimension
        combined_fmaps = torch.cat([fmap1, fmap2], dim=1)  # Concatenate on channel dimension
        combined_fmaps = combined_fmaps.permute(2, 0, 1)  # Reshape for sequence model: (sequence_length, batch_size, embedding_dim)
        
        # Process inertial information with LSTM
        _, (hidden_inertial, _) = self.lstm_inertial(imu)
        # print(hidden_inertial.shape)

        concatenated_features = torch.cat([combined_fmaps, hidden_inertial], dim=0)
        # Cross-attention
        attn_output, _ = self.cross_attention(concatenated_features, concatenated_features, concatenated_features)
        
        attn_output = attn_output.permute(1, 2, 0).reshape(-1, 256, 20, 31)
        attn_output = self.conv_attn_map(attn_output)

        # Flatten the output from attention
        attn_output = attn_output.contiguous().view(attn_output.size(0), -1)
        
        # Position pathway
        pos = self.relu(self.fc1_position(attn_output))
        pos = self.fc2_position(pos)

        # Orientation pathway
        orient = self.relu(self.fc1_orientation(attn_output))
        orient = self.fc2_orientation(orient)
        orient = A_vec_to_quat(orient).reshape(-1, 4)
        
        # print(pos.shape, orient.shape)
        # Combine position and orientation into a single output tensor
        res = torch.cat([pos, orient], dim=1)
    
        return res
    
    def freeze_fnet(self):
        """Freeze the fnet layer to prevent updates during training."""
        for param in self.fnet.parameters():
            param.requires_grad = False
    
    def custom_loss(self, y_hat, y):
        pos_hat, orient_hat = y_hat[:, :3], y_hat[:, 3:]
        pos, orient = y[:, :3], y[:, 3:]
        
        # print(y.shape, y_hat.shape)
        orient_hat = torch.nn.functional.normalize(orient_hat, p=2, dim=1)
        orient = torch.nn.functional.normalize(orient, p=2, dim=1)
        
        # print(pos_hat.shape, pos.shape, orient_hat.shape, orient.shape) #
        
        pos_loss = self.mse(pos_hat, pos)
        # orient_loss = self.geodesic(quaternion_to_matrix(orient_hat), quaternion_to_matrix(orient))
        orient_loss = quat_chordal_squared_loss(orient_hat, orient)
        
        return pos_loss + orient_loss
        
    def training_step(self, batch, batch_idx):
        x, imu, y = batch
        x1, x2 = x[:, 0], x[:, 1]
        y_hat = self(x1, x2, imu)
        loss = self.custom_loss(y_hat, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, imu, y = batch
        x1, x2 = x[:, 0], x[:, 1]
        y_hat = self(x1, x2, imu)
        loss = self.custom_loss(y_hat, y)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, imu, y = batch
        x1, x2 = x[:, 0], x[:, 1]
        y_hat = self(x1, x2, imu)
        loss = self.criterion(y_hat, y)
        self.log('test_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer



class VisualInertialOdometry3(pl.LightningModule):
    def __init__(self, input_dim_inertial=6, hidden_dim_inertial=256, num_layers=2, memory_size=100, dropout=0.5, learning_rate=1e-3):
        super().__init__()
        self.fnet = nn.Sequential(
            BasicConvEncoder(output_dim=256, norm_fn='instance', dropout=dropout),
            POLAUpdate(embed_dim=256, depth=6, num_head=8, window_size=7, neig_win_num=1)
        )
        self.hidden_dim_inertial = hidden_dim_inertial 
        self.conv_reduce = nn.Conv2d(256, 128, kernel_size=1)
        
        self.adaptive_max_pool = nn.AdaptiveMaxPool2d((20, 30))
        self.relu = nn.ReLU()
        
        # self.fmap_transform = nn.Linear(32, 256)
        
        # Updated MultiheadAttention with External Memory
        self.cross_attention = nn.MultiheadAttention(embed_dim=256, num_heads=8, dropout=dropout)

        # LSTM for processing inertial information
        self.lstm_inertial = nn.LSTM(input_size=input_dim_inertial,
                            hidden_size=hidden_dim_inertial,
                            num_layers=num_layers,
                            dropout=dropout,
                            bidirectional=True,
                            batch_first=True)

        
        self.fc1_position = nn.Linear(256, 256)
        self.fc2_position = nn.Linear(256, 3)
        self.fc1_orientation = nn.Linear(256, 256)
        self.fc2_orientation = nn.Linear(256, 10)
        
        self.geodesic = GeodesicLoss()
        self.mse = nn.L1Loss()

    def forward(self, image1, image2, imu):
        image1 = 2 * (image1 / 255.0) - 1.0
        image2 = 2 * (image2 / 255.0) - 1.0
        
        fmap1, fmap2 = self.fnet([image1, image2])
        # print(fmap1.shape, fmap2.shape)
        
        # Reduce dimensionality
        fmap1 = self.conv_reduce(self.adaptive_max_pool(fmap1))
        fmap2 = self.conv_reduce(self.adaptive_max_pool(fmap2))

        # Compute sequence length from spatial dimensions for concatenation
        sequence_length = fmap1.shape[2] * fmap1.shape[3]  # width * height
        fmap1 = fmap1.view(fmap1.shape[0], fmap1.shape[1], sequence_length)
        fmap2 = fmap2.view(fmap2.shape[0], fmap2.shape[1], sequence_length)

        # Concatenate feature maps along the channel dimension
        combined_fmaps = torch.cat([fmap1, fmap2], dim=1)  # Concatenate on channel dimension
        combined_fmaps = combined_fmaps.permute(2, 0, 1)  # Reshape for sequence model: (sequence_length, batch_size, embedding_dim)
        
        # Process inertial information with LSTM
        _, (hidden_inertial, _) = self.lstm_inertial(imu)
        # print(hidden_inertial.shape)

        concatenated_features = torch.cat([combined_fmaps, hidden_inertial], dim=0)
        # Cross-attention
        # attn_output, _ = self.cross_attention(self.fmap_transform(combined_fmaps), inertial_features, inertial_features)
        attn_output, _ = self.cross_attention(concatenated_features, concatenated_features, concatenated_features)
        attn_output = attn_output.mean(dim=0)  # Optionally, reduce along the sequence dimension


        # Flatten the output from attention
        attn_output = attn_output.contiguous().view(attn_output.size(0), -1)
        
        # print(attn_output.shape)

        # Position pathway
        pos = self.relu(self.fc1_position(attn_output))
        pos = self.fc2_position(pos)

        # Orientation pathway
        orient = self.relu(self.fc1_orientation(attn_output))
        orient = self.fc2_orientation(orient)
        orient = A_vec_to_quat(orient).reshape(-1, 4)
        
        # print(pos.shape, orient.shape)
        # Combine position and orientation into a single output tensor
        res = torch.cat([pos, orient], dim=1)
    
        return res
    
    def freeze_fnet(self):
        """Freeze the fnet layer to prevent updates during training."""
        for param in self.fnet.parameters():
            # print(param)
            param.requires_grad = False
    
    def custom_loss(self, y_hat, y):
        pos_hat, orient_hat = y_hat[:, :3], y_hat[:, 3:]
        pos, orient = y[:, :3], y[:, 3:]
        
        # print(y.shape, y_hat.shape)
        orient_hat = torch.nn.functional.normalize(orient_hat, p=2, dim=1)
        orient = torch.nn.functional.normalize(orient, p=2, dim=1)
        
        # print(pos_hat.shape, pos.shape, orient_hat.shape, orient.shape) #
        
        pos_loss = self.mse(pos_hat, pos)
        # orient_loss = self.geodesic(quaternion_to_matrix(orient_hat[:, [-1, 0, 1, 2]]), quaternion_to_matrix(orient[:, [-1, 0, 1, 2]]))
        orient_loss = quat_chordal_squared_loss(orient_hat, orient)
        
        return pos_loss + orient_loss
        
    def training_step(self, batch, batch_idx):
        x, imu, y = batch
        x1, x2 = x[:, 0], x[:, 1]
        y_hat = self(x1, x2, imu)
        loss = self.custom_loss(y_hat, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, imu, y = batch
        x1, x2 = x[:, 0], x[:, 1]
        y_hat = self(x1, x2, imu)
        loss = self.custom_loss(y_hat, y)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, imu, y = batch
        x1, x2 = x[:, 0], x[:, 1]
        y_hat = self(x1, x2, imu)
        loss = self.criterion(y_hat, y)
        self.log('test_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss
    

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer
