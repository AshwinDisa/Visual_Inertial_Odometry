import os
import argparse
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, ModelSummary
# from utils.data_module import TrajectoryDataModule
from models.io import BidirectionalLSTM
from models.vo import VisualOdometry
from models.vio import VisualInertialOdometry, VisualInertialOdometry3

from models.losses import quaternion_to_matrix
from tqdm import tqdm
import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial.transform import Rotation

from pytorch_lightning import LightningDataModule
import torch.utils.data as data
from torch.utils.data import DataLoader
from datasets.dataset import MultiTrajectoryDataset
import torch
import transformations as tf

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def logmap_so3(R):
    theta = np.arccos((np.trace(R) - 1) / 2)
    return theta / (2 * np.sin(theta)) * (R - R.T)

def compute_absolute_error(p_es_aligned, q_es_aligned, p_gt, q_gt):
    e_trans_vec = (p_gt-p_es_aligned)
    e_trans = np.sqrt(np.sum(e_trans_vec**2, 1))

    # orientation error
    e_rot = np.zeros((len(e_trans,)))
    e_ypr = np.zeros(np.shape(p_es_aligned))
    for i in range(np.shape(p_es_aligned)[0]):
        R_we = tf.quaternion_matrix(q_es_aligned[i, :])
        R_wg = tf.quaternion_matrix(q_gt[i, :])
        e_R = np.dot(R_we, np.linalg.inv(R_wg))
        e_ypr[i, :] = tf.euler_from_matrix(e_R, 'rzyx')
        e_rot[i] = np.rad2deg(np.linalg.norm(logmap_so3(e_R[:3, :3])))

    # scale drift
    motion_gt = np.diff(p_gt, 0)
    motion_es = np.diff(p_es_aligned, 0)
    dist_gt = np.sqrt(np.sum(np.multiply(motion_gt, motion_gt), 1))
    dist_es = np.sqrt(np.sum(np.multiply(motion_es, motion_es), 1))
    e_scale_perc = np.abs((np.divide(dist_es, dist_gt)-1.0) * 100)

    return e_trans, e_trans_vec, e_rot, e_ypr, e_scale_perc

def compute_ate(gt_positions, pred_positions):
    # Compute the Euclidean distance between ground truth and predicted positions
    distances = np.linalg.norm(gt_positions - pred_positions, axis=1)
    return distances


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
    
    if args.mode == 'VO':
        #VO
        model_dir = f"trained_weights/final_geodesic_VO"
        checkpoint_path = os.path.join(model_dir, '31-0.00695.ckpt')
        
    if args.mode == 'IO':
        #Io
        model_dir = f"trained_weights/final_geodesic_IO/Geodesic/"
        checkpoint_path = os.path.join(model_dir, '186-0.00484.ckpt')
        
    if args.mode == 'VIO':
        #VIO
        model_dir = f"trained_weights/final_geodesic_VIO/Bingham_new"
        checkpoint_path = os.path.join(model_dir, '27-0.02476.ckpt')
        
        
    # Initialize the Data Module
    data_module = TrajectoryDataModule(json_file=args.data_file, batch_size=args.batch_size, mode=args.mode)

    # Select the model based on the mode
    if args.mode == 'IO':
        model = BidirectionalLSTM.load_from_checkpoint(checkpoint_path)
    elif args.mode == 'VO':
        model = VisualOdometry.load_from_checkpoint(checkpoint_path)
    elif args.mode == 'VIO':
        model = VisualInertialOdometry3.load_from_checkpoint(checkpoint_path)
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
            
            if args.mode == 'IO':
                x, y = batch  # Assuming the dataloader returns pairs
                y_hat = model(x.to(device))  # Output: [xyz, quaternion]
                # model.to_onnx("IO.onnx", x, export_params=True)
            
            elif args.mode == 'VO':
                x, y = batch  # Assuming the dataloader returns pairs
                x1, x2 = x[:, 0].to(device), x[:, 1].to(device)
                y_hat = model(x1, x2)  # Output: [xyz, quaternion]
                
            elif args.mode == 'VIO':
                x, imu, y = batch
                x1, x2 = x[:, 0].to(device), x[:, 1].to(device)
                y_hat = model(x1, x2, imu.to(device))
                
                
            pos_hat, orient_hat = y_hat[:, :3], y_hat[:, 3:]
            R = quaternion_to_matrix(orient_hat)
            
            
            pred_position.append(pos_hat.detach().cpu().numpy())
            pred_orientation.append(R.detach().cpu().numpy())
            
            
            gt_pos, gt_orient = y[:, :3].to(device), y[:, 3:].to(device)
            gt_R = quaternion_to_matrix(gt_orient)
            
            gt_position.append(gt_pos.detach().cpu().numpy())
            gt_orientation.append(gt_R.detach().cpu().numpy())

    
    # Integrate the relative poses to get the absolute trajectory
    gt_abs_positions, gt_abs_orientations = integrate_trajectory(gt_position, gt_orientation)
    abs_positions, abs_orientations = integrate_trajectory(pred_position, pred_orientation)
    print(abs_positions.shape, len(gt_abs_positions))
    print(f"Gt: {gt_abs_positions[0]} Pred: {abs_positions[0]}")
    
    def rotation_matrices_to_euler(rot_matrices):
        """Convert a series of rotation matrices to Euler angles."""
        return Rotation.from_matrix(rot_matrices).as_euler('xyz', degrees=True)  # returns angles in degrees

    # Convert ground truth and predicted orientation matrices to Euler angles
    gt_euler_angles = rotation_matrices_to_euler(gt_abs_orientations)
    pred_euler_angles = rotation_matrices_to_euler(abs_orientations)
    
    # Compute ATE distances
    ate_distances = compute_ate(gt_abs_positions.squeeze(), abs_positions.squeeze())

    # Compute the mean and median of ATE
    ate_mean = np.mean(ate_distances)
    ate_median = np.median(ate_distances)

    print(f"Mean ATE: {ate_mean:.3f} meters")
    print(f"Median ATE: {ate_median:.3f} meters")
    
    
    def to_4x4_matrix(rot_matrix):
        mat = np.eye(4)
        mat[:3, :3] = rot_matrix
        return mat

    q_es_aligned = np.array([tf.quaternion_from_matrix(to_4x4_matrix(mat)) for mat in abs_orientations])
    q_gt = np.array([tf.quaternion_from_matrix(to_4x4_matrix(mat)) for mat in gt_abs_orientations])

    # Compute errors:
    e_trans, e_trans_vec, e_rot, e_ypr, e_scale_perc = compute_absolute_error(abs_positions.squeeze(), q_es_aligned, gt_abs_positions.squeeze(), q_gt)

    # Now, you can use these errors for further analysis or reporting.
    print(f"Mean Translation Error: {np.mean(e_trans):.3f} meters")
    print(f"Mean Rotation Error: {np.mean(e_rot):.3f} degrees")
    print(f"Mean Scale Drift Percentage: {np.mean(e_scale_perc):.3f}%")
        
    
    fig = plt.figure(figsize=(24, 18))  # Adjust the figure size to better accommodate the additional plot

    # 3D trajectory plot
    ax1 = fig.add_subplot(331, projection='3d')  # Adjust subplot positioning
    ax1.plot(abs_positions[:, 0], abs_positions[:, 1], abs_positions[:, 2], 'b-', label='Estimated Trajectory')
    ax1.plot(gt_abs_positions[:, 0], gt_abs_positions[:, 1], gt_abs_positions[:, 2], 'r-', label='Ground Truth Trajectory')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.set_title('3D Trajectory Plot')
    ax1.legend()

    # XY plane plot
    ax2 = fig.add_subplot(332)  # Adjust subplot positioning
    ax2.plot(abs_positions[:, 0], abs_positions[:, 1], 'b-', label='Estimated Trajectory')
    ax2.plot(gt_abs_positions[:, 0], gt_abs_positions[:, 1], 'r-', label='Ground Truth Trajectory')
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_title('XY Plane Trajectory')
    ax2.legend()

    # XZ plane plot
    ax3 = fig.add_subplot(333)  # Adjust subplot positioning
    ax3.plot(abs_positions[:, 0], abs_positions[:, 2], 'b-', label='Estimated Trajectory')
    ax3.plot(gt_abs_positions[:, 0], gt_abs_positions[:, 2], 'r-', label='Ground Truth Trajectory')
    ax3.set_xlabel('X (m)')
    ax3.set_ylabel('Z (m)')
    ax3.set_title('XZ Plane Trajectory')
    ax3.legend()

    # YZ plane plot
    ax7 = fig.add_subplot(334)  # New subplot for YZ plane
    ax7.plot(abs_positions[:, 1], abs_positions[:, 2], 'b-', label='Estimated Trajectory')
    ax7.plot(gt_abs_positions[:, 1], gt_abs_positions[:, 2], 'r-', label='Ground Truth Trajectory')
    ax7.set_xlabel('Y (m)')
    ax7.set_ylabel('Z (m)')
    ax7.set_title('YZ Plane Trajectory')
    ax7.legend()

    # Roll plot
    ax4 = fig.add_subplot(335)  # Adjust subplot positioning
    ax4.plot(pred_euler_angles[:, 0], 'b-', label='Estimated Roll')
    ax4.plot(gt_euler_angles[:, 0], 'r-', label='Ground Truth Roll')
    ax4.set_xlabel('Time')
    ax4.set_ylabel('Degrees')
    ax4.set_title('Roll')
    ax4.legend()

    # Pitch plot
    ax5 = fig.add_subplot(336)  # Adjust subplot positioning
    ax5.plot(pred_euler_angles[:, 1], 'b-', label='Estimated Pitch')
    ax5.plot(gt_euler_angles[:, 1], 'r-', label='Ground Truth Pitch')
    ax5.set_xlabel('Time')
    ax5.set_ylabel('Degrees')
    ax5.set_title('Pitch')
    ax5.legend()

    # Yaw plot
    ax6 = fig.add_subplot(337)  # Adjust subplot positioning
    ax6.plot(pred_euler_angles[:, 2], 'b-', label='Estimated Yaw')
    ax6.plot(gt_euler_angles[:, 2], 'r-', label='Ground Truth Yaw')
    ax6.set_xlabel('Time')
    ax6.set_ylabel('Degrees')
    ax6.set_title('Yaw')
    ax6.legend()

    plt.tight_layout()  # Adjust the layout to make room for all subplots
    plt.savefig('visualizations/trajectory_and_orientation_plot.png') 
    
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate models for odometry estimation")
    parser.add_argument('--data_file', type=str, required=True, help='Path to the data file')
    parser.add_argument('--batch_size', type=int, default=1, help='Input batch size for training')
    parser.add_argument('--mode', type=str, choices=['VO', 'IO', 'VIO'], required=True, help='Mode of operation: VO')
    parser.add_argument('--gpus', type=int, default=1, help='Number of GPUs to use')
    args = parser.parse_args()
    main(args)
