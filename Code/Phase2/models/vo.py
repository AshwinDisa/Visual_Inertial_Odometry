import torch
from torch import nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from models.helpers import BasicConvEncoder
from models.helpers import POLAUpdate
from models.helpers import DropPath
from models.helpers import to_2tuple
from models.losses import convert_Avec_to_A, A_vec_to_quat
from models.losses import quat_consistency_loss, quat_chordal_squared_loss
from models.losses import GeodesicLoss


class VisualOdometry(pl.LightningModule):
    def __init__(self, dropout=0.0, learning_rate=1e-3):
        super().__init__()
        self.fnet = nn.Sequential(
            BasicConvEncoder(output_dim=256, norm_fn='instance', dropout=dropout),
            POLAUpdate(embed_dim=256, depth=6, num_head=8, window_size=7, neig_win_num=1)
        )
        self.conv1_position = nn.Conv2d(512, 16, kernel_size=1)
        self.conv1_orientation = nn.Conv2d(512, 16, kernel_size=1)
        
        self.adaptive_max_pool = nn.AdaptiveMaxPool2d((30, 40))
        self.relu = nn.ReLU()
        
        self.fc1_position = nn.Linear(19200, 256)
        self.fc2_position = nn.Linear(256, 3)
        self.fc1_orientation = nn.Linear(19200, 256)
        self.fc2_orientation = nn.Linear(256, 4)
        
        self.geodesic = GeodesicLoss()
        self.mse = nn.L1Loss()

    def forward(self, image1, image2):
        image1 = 2 * (image1 / 255.0) - 1.0
        image2 = 2 * (image2 / 255.0) - 1.0

        fmap1, fmap2 = self.fnet([image1, image2])
        # print(fmap1.shape, fmap2.shape) #
        feats = torch.cat([fmap1, fmap2], dim=1)
        
        # Position pathway
        pos = self.conv1_position(feats)
        pos = self.adaptive_max_pool(pos)
        
        pos = pos.view(pos.size(0), -1)  # Flatten
        pos = self.relu(self.fc1_position(pos))
        pos = self.fc2_position(pos)

        # Orientation pathway
        orient = self.conv1_orientation(feats)
        orient = self.adaptive_max_pool(orient)
        
        orient = orient.view(orient.size(0), -1)  # Flatten
        orient = self.relu(self.fc1_orientation(orient))
        orient = self.fc2_orientation(orient)
        
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
        
        orient_hat = torch.nn.functional.normalize(orient_hat, p=2, dim=1)
        orient = torch.nn.functional.normalize(orient, p=2, dim=1)
        
        pos_loss = self.mse(pos_hat, pos)
        orient_loss = self.geodesic(quaternion_to_matrix(orient_hat), quaternion_to_matrix(orient))
        # orient_loss = quat_chordal_squared_loss(orient_hat, orient)
        
        return pos_loss + orient_loss
        
    def training_step(self, batch, batch_idx):
        x, y = batch
        x1, x2 = x[:, 0], x[:, 1]
        y_hat = self(x1, x2)
        loss = self.custom_loss(y_hat, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x1, x2 = x[:, 0], x[:, 1]
        y_hat = self(x1, x2)
        loss = self.custom_loss(y_hat, y)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        x1, x2 = x[:, 0], x[:, 1]
        y_hat = self(x1, x2)
        loss = self.criterion(y_hat, y)
        self.log('test_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer
    
    
    
    
    
    

class VisualOdometry_Attn(pl.LightningModule):
    def __init__(self, dropout=0.0, learning_rate=1e-3):
        super().__init__()
        self.fnet = nn.Sequential(
            BasicConvEncoder(output_dim=256, norm_fn='instance', dropout=dropout),
            POLAUpdate(embed_dim=256, depth=6, num_head=8, window_size=7, neig_win_num=1)
        )
        self.conv1_position = nn.Conv2d(512, 16, kernel_size=1)
        # self.conv2_position = nn.Conv2d(256, 16, kernel_size=1)
        
        self.conv1_orientation = nn.Conv2d(512, 16, kernel_size=1)
        # self.conv2_orientation = nn.Conv2d(256, 16, kernel_size=1)
        
        self.adaptive_max_pool = nn.AdaptiveMaxPool2d((30, 40))
        self.relu = nn.ReLU()
        
        self.fc1_position = nn.Linear(19200, 256)
        self.fc2_position = nn.Linear(256, 3)
        self.fc1_orientation = nn.Linear(19200, 256)
        self.fc2_orientation = nn.Linear(256, 10)
        
        self.geodesic = GeodesicLoss()
        self.mse = nn.L1Loss()

    def forward(self, image1, image2):
        image1 = 2 * (image1 / 255.0) - 1.0
        image2 = 2 * (image2 / 255.0) - 1.0

        fmap1, fmap2 = self.fnet([image1, image2])
        # print(fmap1.shape, fmap2.shape) #
        feats = torch.cat([fmap1, fmap2], dim=1)
        # print(feats.shape) #
        
        # Position pathway
        pos = self.conv1_position(feats)
        # pos = self.conv2_position(pos)
        
        pos = self.adaptive_max_pool(pos)
        
        pos = pos.view(pos.size(0), -1)  # Flatten
        pos = self.relu(self.fc1_position(pos))
        pos = self.fc2_position(pos)

        # Orientation pathway
        orient = self.conv1_orientation(feats)
        # orient = self.conv2_orientation(orient)
        
        orient = self.adaptive_max_pool(orient)
        
        orient = orient.view(orient.size(0), -1)  # Flatten
        orient = self.relu(self.fc1_orientation(orient))
        orient = self.fc2_orientation(orient)
        
        # print(pos.shape, orient.shape[1]) #
        
        orient = A_vec_to_quat(orient).reshape(-1, 4)
        
        # print(f"Shape: {orient.shape}")
        # print(f"Pos shape: {pos.shape}")
        
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
        x, y = batch
        x1, x2 = x[:, 0], x[:, 1]
        y_hat = self(x1, x2)
        loss = self.custom_loss(y_hat, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x1, x2 = x[:, 0], x[:, 1]
        y_hat = self(x1, x2)
        loss = self.custom_loss(y_hat, y)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        x1, x2 = x[:, 0], x[:, 1]
        y_hat = self(x1, x2)
        loss = self.criterion(y_hat, y)
        self.log('test_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss
    

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer

