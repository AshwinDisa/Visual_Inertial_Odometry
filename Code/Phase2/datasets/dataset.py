import torch
from torch.utils.data import Dataset, DataLoader
import json
import pandas as pd
import random
import cv2
import torch
import json
import pandas as pd
import random
from torch.utils.data import Dataset, DataLoader
import numpy as np

class MultiTrajectoryDataset(Dataset):
    def __init__(self, main_json_file, mode='VO', train=True):
        print("Loading dataset...")
        print("Mode: ", mode)
        print("Main JSON file: ", main_json_file)
        self.main_data = json.load(open(main_json_file))
        self.mode = mode
        self.trajectories = self.load_trajectories()
        self.train = train

    def load_trajectories(self):
        trajectories = []
        for traj in self.main_data['trajectories']:
            data_entries = json.load(open(traj['data_entries']))
            # print("Data entries: ", data_entries['timestamp'])
            imu_data = pd.read_csv(traj['imu_readings'])
            poses = pd.read_csv(traj['poses'])
            trajectories.append({
                'trajectory_id': traj['trajectory_id'],
                'data_entries': data_entries,
                'imu_data': imu_data,
                'poses': poses
            })
        return trajectories

    def __len__(self):
        return sum(len(traj['data_entries']['entries']) for traj in self.trajectories)
    
    def __getitem__(self, idx):
        # Randomly select a trajectory
        
        chosen_trajectory = random.choice(self.trajectories)
        # chosen_trajectory = self.trajectories[0]
        # print("Chosen trajectory: ", chosen_trajectory)
            
        trajectory_id = chosen_trajectory['trajectory_id']
        # Randomly select an entry from the chosen trajectory
        if self.train:
            entry = random.choice(chosen_trajectory['data_entries']['entries'])
            # entry = chosen_trajectory['data_entries']['entries'][0]
        else:
            entry = chosen_trajectory['data_entries']['entries'][idx]
           
        # print("Entry: ", entry)
        pose = self.load_pose(entry['pose_index'], chosen_trajectory['poses'])
        # print("Pose: ", pose)
        # print("entry: ", entry)
        
        if self.mode == 'VO':
            # print(f"{entry['images']}")
            images = self.load_images(entry['images'], trajectory_id)
            return images, pose
        elif self.mode == 'IO':
            # print(trajectory_id)
            imu_data = self.load_imu_data(entry['imu_start_index'], entry['imu_end_index'], chosen_trajectory['imu_data'])
            # print("IMU data: ", imu_data)
            # print("Pose: ", pose)
            return imu_data, pose
        elif self.mode == 'VIO':
            images = self.load_images(entry['images'], trajectory_id)
            imu_data = self.load_imu_data(entry['imu_start_index'], entry['imu_end_index'], chosen_trajectory['imu_data'])
            return images, imu_data, pose
    
    def load_images(self, image_files, trajectory_id):
        # Assuming image loading returns a te"nsor
        # print("Image files: ", image_files)
        replace_with = 'trajectory' + trajectory_id.split('_')[1]
        cv2_images = [cv2.imread(f"data/Trajectories/{trajectory_id}/{replace_with}/DownCam/{img_file}") for img_file in image_files]
        # print(torch.tensor(cv2_images, dtype=torch.float32).permute(0, 3, 1, 2).shape)
        return torch.tensor(np.array(cv2_images), dtype=torch.float32).permute(0, 3, 1, 2)  # Convert list of images to tensor
    
    def load_imu_data(self, start_index, end_index, imu_data):
        # Extract and preprocess IMU data for the given range
        # print("Start index: ", start_index , "End index: ", end_index)
        imu_slice = imu_data.iloc[start_index-1:end_index]
        # Exclude the first column using .iloc[:, 1:] to select all rows and columns from the second to the last
        imu_slice = imu_slice.iloc[:, 1:]
        # print("slice.shape: ", imu_slice.shape)
        return torch.tensor(imu_slice.values, dtype=torch.float32)  # Convert DataFrame to tensor
    
    def load_pose(self, pose_index, poses):
        # Extract pose information for the given index
        pose = poses.iloc[pose_index]
        pose = pose.iloc[1:]
        return torch.tensor(pose.values, dtype=torch.float32)  # Convert DataFrame to tensor

# # Example usage
# dataset = MultiTrajectoryDataset('main_data.json', mode='VIO')
# dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
