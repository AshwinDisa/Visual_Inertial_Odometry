import os
import argparse
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, ModelSummary
# from utils.data_module import TrajectoryDataModule
from models.io import BidirectionalLSTM
from models.vo import VisualOdometry
from models.vio import VisualInertialOdometry, VisualInertialOdometry2

from pytorch_lightning import LightningDataModule
import torch.utils.data as data
from torch.utils.data import DataLoader
from datasets.dataset import MultiTrajectoryDataset
import torch


class TrajectoryDataModule(LightningDataModule):
    def __init__(self, json_file, batch_size=32, mode='VIO', num_workers=2):
        super().__init__()
        self.json_file = json_file
        self.batch_size = batch_size
        self.mode = mode
        self.num_workers = num_workers

    def setup(self, stage=None):
        # Called on each GPU separately
        # Setup data specifics (train, val, test split, etc.)
        # For simplicity, we're using the same dataset for training and validation
        print(f"Setting up data for {self.mode} mode")
        dataset = MultiTrajectoryDataset(self.json_file, mode=self.mode)
        train_set_size = int(len(dataset) * 0.8)
        valid_set_size = len(dataset) - train_set_size

        # split the train set into two
        seed = torch.Generator().manual_seed(42)
        self.train_set, self.valid_set = data.random_split(dataset, [train_set_size, valid_set_size], generator=seed)

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        # Optionally implement this if you have a validation dataset
        return DataLoader(self.valid_set, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        # Optionally implement this if you have a test dataset
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)



def main(args):
    # Create a directory for trained weights if it doesn't exist
    model_dir = f"trained_weights/final_geodesic_{args.mode}"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # Initialize the Data Module
    data_module = TrajectoryDataModule(json_file=args.data_file, batch_size=args.batch_size, mode=args.mode)

    # Select the model based on the mode
    if args.mode == 'IO':
        model = BidirectionalLSTM(input_dim=6, hidden_dim=256, output_dim=7, num_layers=2)
        
    elif args.mode == 'VO':
        model = VisualOdometry()
        weights = torch.load("pretrained/gmflownet-kitti.pth", map_location='cpu')
        
        for key in weights.keys():
            if key.replace('module.', '') in model.state_dict().keys():
                model.state_dict()[key.replace('module.', '')] = weights[key]
                
        model.freeze_fnet()  # Freeze fnet after loading the model
        
    elif args.mode == 'VIO':
        model = VisualInertialOdometry2(input_dim_inertial=6, hidden_dim_inertial=256, num_layers=2)
        weights = torch.load("pretrained/gmflownet-kitti.pth", map_location='cpu')
        lstm_weights = torch.load("trained_weights/final_geodesic_IO/Geodesic/186-0.00484.ckpt", map_location='cpu')['state_dict']
        
        for key in weights.keys():
            if key.replace('module.', '') in model.state_dict().keys():
                model.state_dict()[key.replace('module.', '')] = weights[key]
           
        for key in lstm_weights.keys():
            if key.replace('lstm.', 'lstm_inertial.') in model.state_dict().keys():
                model.state_dict()[key.replace('lstm.', 'lstm_inertial.')] = lstm_weights[key]     
          
        # model.freeze_fnet()  # Freeze fnet after loading the model
        
    else:
        raise ValueError("Unsupported mode! Use 'IO' for inertial odometry.")

    # Logger setup
    logger = TensorBoardLogger("tb_logs", name=f"my_model_{args.mode}")

    # Checkpoint callback to save the best model based on validation loss
    checkpoint_callback = ModelCheckpoint(
        dirpath=model_dir,
        filename='{epoch}-{val_loss:.5f}',
        save_top_k=3,  # saves only the best model
        monitor='val_loss',  # metric to monitor
        mode='min',  # mode of the monitored quantity for optimization
        auto_insert_metric_name=False  # prevent automatic insertion of the metric name in the filename
    )

    # Trainer setup
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        logger=logger,
        callbacks=[checkpoint_callback],
        precision=16 if args.fp16 else 32,
        devices=args.gpus,
        log_every_n_steps=10
    )
    
    # Train the model
    trainer.fit(model, data_module)
 

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train models for different modalities")
    parser.add_argument('--data_file', type=str, required=True, help='Path to the data file')
    parser.add_argument('--batch_size', type=int, default=32, help='Input batch size for training')
    parser.add_argument('--mode', type=str, choices=['VO', 'IO', 'VIO'], required=True, help='Mode of operation: VO, IO, or VIO')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs to train')
    parser.add_argument('--gpus', type=int, default=1, help='Number of GPUs')
    parser.add_argument('--fp16', action='store_true', help='Enable FP16 training')

    args = parser.parse_args()
    main(args)