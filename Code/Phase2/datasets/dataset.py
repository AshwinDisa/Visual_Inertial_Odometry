import torch
from torch.utils.data import Dataset, DataLoader
import json
import pandas as pd
import random

class MultiTrajectoryDataset(Dataset):
    def __init__(self, main_json_file, mode='VO'):
        self.main_data = json.load(open(main_json_file))
        self.mode = mode
        self.trajectories = self.load_trajectories()

    def load_trajectories(self):
        trajectories = []
        for traj in self.main_data['trajectories']:
            data_entries = json.load(open(traj['data_entries']))
            imu_data = pd.read_csv(traj['imu_readings'])
            poses = pd.read_csv(traj['poses'])
            trajectories.append({
                'data_entries': data_entries,
                'imu_data': imu_data,
                'poses': poses
            })
        return trajectories

    def __len__(self):
        # This could be more sophisticated depending on how you want to sample
        return sum(len(traj['data_entries']['entries']) for traj in self.trajectories)
    
    def __getitem__(self, idx):
        # Randomly select a trajectory
        chosen_trajectory = random.choice(self.trajectories)
        # Randomly select an entry from the chosen trajectory
        entry = random.choice(chosen_trajectory['data_entries']['entries'])
        
        if self.mode == 'VO':
            images = self.load_images(entry['images'])
            pose = self.load_pose(entry['pose_index'], chosen_trajectory['poses'])
            return images, pose
        elif self.mode == 'IO':
            imu_data = self.load_imu_data(entry['imu_start_index'], entry['imu_end_index'], chosen_trajectory['imu_data'])
            return imu_data
        elif self.mode == 'VIO':
            images = self.load_images(entry['images'])
            imu_data = self.load_imu_data(entry['imu_start_index'], entry['imu_end_index'], chosen_trajectory['imu_data'])
            pose = self.load_pose(entry['pose_index'], chosen_trajectory['poses'])
            return images, imu_data, pose
    
    def load_images(self, image_files):
        # Load and preprocess images
        pass
    
    def load_imu_data(self, start_index, end_index, imu_data):
        # Extract and preprocess IMU data for the given range
        return imu_data.iloc[start_index-1:end_index]
    
    def load_pose(self, pose_index, poses):
        # Extract pose information for the given index
        return poses.iloc[pose_index - 1]

# Example usage
dataset = MultiTrajectoryDataset('main_data.json', mode='VIO')
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
