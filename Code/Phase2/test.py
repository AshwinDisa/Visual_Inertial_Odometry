import os
import argparse
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, ModelSummary
# from utils.data_module import TrajectoryDataModule
from models.io import BidirectionalLSTM
from models.vo import VisualOdometry, quaternion_to_matrix
from tqdm import tqdm
import numpy as np
from matplotlib import pyplot as plt

from pytorch_lightning import LightningDataModule
import torch.utils.data as data
from torch.utils.data import DataLoader
from datasets.dataset import MultiTrajectoryDataset
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def integrate_trajectory(rel_positions, rel_orientations):
    n_samples = len(rel_positions)
    abs_positions = [np.zeros((3, 1))]*n_samples  # Assuming batch size is handled inside model
    abs_orientations = [np.eye(3)] * n_samples
    
    for i in range(1, n_samples):
        
        abs_orientations[i] = np.dot(abs_orientations[i-1], rel_orientations[i].reshape(3, 3))
        abs_positions[i] = abs_positions[i-1].reshape(3, 1) + np.dot(rel_orientations[i].reshape(3, 3), rel_positions[i].reshape(3, 1))
    # print("2")  
    return np.array(abs_positions), abs_orientations


class TrajectoryDataModule(LightningDataModule):
    def __init__(self, json_file, batch_size=1, mode='VIO', num_workers=1):
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
        dataset = MultiTrajectoryDataset(self.json_file, mode=self.mode, train=False)
        train_set_size = int(len(dataset) * 0.8)
        valid_set_size = len(dataset) - train_set_size

        # split the train set into two
        seed = torch.Generator().manual_seed(42)
        # self.train_set, self.valid_set = data.random_split(dataset, [train_set_size, valid_set_size], generator=seed)
        self.train_set = dataset

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        # Optionally implement this if you have a validation dataset
        return DataLoader(self.valid_set, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        # Optionally implement this if you have a test dataset
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)


def main(args):
    # Create a directory for trained weights if it doesn't exist
    # model_dir = f"trained_weights/{args.mode}"
    model_dir = f"trained_weights/freezed_VO"
    
    checkpoint_path = os.path.join(model_dir, '17-0.00389.ckpt')
    
    
    # Initialize the Data Module
    data_module = TrajectoryDataModule(json_file=args.data_file, batch_size=args.batch_size, mode=args.mode)

    # Select the model based on the mode
    if args.mode == 'IO':
        model = BidirectionalLSTM(input_dim=6, hidden_dim=256, output_dim=7, num_layers=2).load_from_checkpoint(checkpoint_path)
    elif args.mode == 'VO':
        model = VisualOdometry.load_from_checkpoint(checkpoint_path)
    else:
        raise ValueError("Unsupported mode! Use IO, VO or VIO")
    
    model.eval()

    data_module = TrajectoryDataModule(json_file=args.data_file, batch_size=args.batch_size, mode=args.mode)
    data_module.setup(stage='test')

    trainer = pl.Trainer(devices=args.gpus)

    test_loader = data_module.test_dataloader()
    current_pose = torch.eye(4).to(device)  # Initial pose as identity matrix
    gt_current_pose = torch.eye(4).to(device)
    pred_position = []  # Start with the initial pose
    pred_orientation = []  # Start with the initial pose
    gt_orientation = []
    gt_position = []
    index = 0
    
    with torch.no_grad():

        for batch in tqdm(test_loader, desc="Processing batches"):
            x, y = batch  # Assuming the dataloader returns pairs
            # x, y = test_loader.dataset[index]
            
            
            x1, x2 = x[:, 0].to(device), x[:, 1].to(device)
            y_hat = model(x1, x2)  # Output: [xyz, quaternion]
            pos_hat, orient_hat = y_hat[:, :3], y_hat[:, 3:]
            # R = quaternion_to_matrix(orient_hat[:, [-1, 0, 1, 2]])  # Reorder quaternion to [w, x, y, z]
            # print(orient_hat)
            # print(orient_hat[:, [-1, 0, 1, 2]])
            # normalize the quaternion
            orient_hat = orient_hat / torch.norm(orient_hat, dim=1, keepdim=True)
            R = quaternion_to_matrix(orient_hat[:, [-1, 0, 1, 2]])  # Reorder quaternion to [w, x, y, z]
            
            
            pred_position.append(pos_hat.detach().cpu().numpy())
            pred_orientation.append(R.detach().cpu().numpy())
            
            
            gt_pos, gt_orient = y[:, :3].to(device), y[:, 3:].to(device)
            gt_orient = gt_orient / torch.norm(gt_orient, dim=1, keepdim=True)
            gt_R = quaternion_to_matrix(gt_orient[:, [-1, 0, 1, 2]])
            
            gt_position.append(gt_pos.detach().cpu().numpy())
            gt_orientation.append(gt_R.detach().cpu().numpy())

    
    # Integrate the relative poses to get the absolute trajectory
    gt_abs_positions, gt_abs_orientations = integrate_trajectory(gt_position, gt_orientation)
    abs_positions, abs_orientations = integrate_trajectory(pred_position, pred_orientation)
    print(abs_positions.shape, len(gt_abs_positions))
    print(f"Gt: {gt_abs_positions[0]} Pred: {abs_positions[0]}")
    

    # Plotting the trajectory
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(abs_positions[:,0], abs_positions[:, 1], abs_positions[:, 2], 'b-', label='Estimated Trajectory')
    ax.plot(gt_abs_positions[:, 0], gt_abs_positions[:, 1], gt_abs_positions[:, 2], 'r-', label='Ground Truth Trajectory')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('3D Trajectory Plot')
    plt.savefig('visualizations/trajectory_plot.png')  # Save the figure
    # plt.show()
    
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate models for odometry estimation")
    parser.add_argument('--data_file', type=str, required=True, help='Path to the data file')
    parser.add_argument('--batch_size', type=int, default=1, help='Input batch size for training')
    parser.add_argument('--mode', type=str, choices=['VO', 'IO', 'VIO'], required=True, help='Mode of operation: VO')
    parser.add_argument('--gpus', type=int, default=1, help='Number of GPUs to use')
    args = parser.parse_args()
    main(args)
