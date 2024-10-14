import torch
from torch.utils.data import TensorDataset
import torch.nn as nn
import numpy as np
import os

class MassMaps(TensorDataset): 
    def __init__(self, data_dir, split='train', download=False):
        if download: 
            raise ValueError("download not implemented")
            
        folder = os.path.join(data_dir, "mass_maps", f"mass_maps_{split}")
        if split == 'train': 
            X,y = [os.path.join(folder, f"{a}_maps_Cosmogrid_100k.npy")
                                for a in ["X", "y"]]
        elif split == 'test': 
            X,y = [os.path.join(folder, f"test_{a}.npy")
                                for a in ["x", "y"]]
        else: 
            raise ValueError("Split should be either train or test")
        
        X,y = [torch.from_numpy(np.load(a)) for a in (X,y)]
        X = X.unsqueeze(1)
        
        super(MassMaps,self).__init__(X,y)

class MassMapsConvnet(nn.Module):
    def __init__(self, data_dir):
        super(MassMapsConvnet, self).__init__()
        output_num = 2
        
        # self.normalization = nn.BatchNorm2d(1)
        self.conv1 = nn.Conv2d(1, 16, kernel_size=4)
        self.relu1 = nn.LeakyReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4)
        self.relu2 = nn.LeakyReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.conv3 = nn.Conv2d(32, 48, kernel_size=4)
        self.relu3 = nn.LeakyReLU()
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(1200, 128)
        self.relu4 = nn.LeakyReLU()
        self.fc2 = nn.Linear(128, 64)
        self.relu5 = nn.LeakyReLU()
        self.fc3 = nn.Linear(64, 32)
        self.relu6 = nn.LeakyReLU()
        self.fc4 = nn.Linear(32, output_num)
        
        weights_path = os.path.join(data_dir, "mass_maps", "mass_maps_pspnet", "CNN_mass_maps.pth")
        self.load_state_dict(torch.load(weights_path))
        
    def forward(self, x):
        # x = self.normalization(x)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.maxpool3(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu4(x)
        x = self.fc2(x)
        x = self.relu5(x)
        x = self.fc3(x)
        x = self.relu6(x)
        x = self.fc4(x)
        return x