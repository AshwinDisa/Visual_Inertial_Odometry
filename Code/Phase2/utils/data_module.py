from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from datasets.multi_trajectory_dataset import MultiTrajectoryDataset

class TrajectoryDataModule(LightningDataModule):
    def __init__(self, json_file, batch_size=32, mode='VIO', num_workers=4):
        super().__init__()
        self.json_file = json_file
        self.batch_size = batch_size
        self.mode = mode
        self.num_workers = num_workers

    def setup(self, stage=None):
        # Called on each GPU separately
        # Setup data specifics (train, val, test split, etc.)
        # For simplicity, we're using the same dataset for training and validation
        self.dataset = MultiTrajectoryDataset(self.json_file, mode=self.mode)

    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        # Optionally implement this if you have a validation dataset
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        # Optionally implement this if you have a test dataset
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
