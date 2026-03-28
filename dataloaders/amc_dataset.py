import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

class RadioMLDataset(Dataset):
    """
    Industrial-grade PyTorch Dataset for RadioML HDF5 datasets.
    Handles lazy loading of H5 files to prevent deadlocks in multi-process DataLoaders.
    """
    def __init__(self, file_path):
        """
        Args:
            file_path (str): Path to the HDF5 file.
            snr_threshold (int/float, optional): Minimum SNR value to include.
        """
        self.file_path = file_path
        self.archive = None

        # Pre-load labels and SNR to memory for indexing and filtering
        with h5py.File(self.file_path, 'r') as f:
            # Y is one-hot encoded (N, 24)
            y_data = f['Y'][:]
            # Z is SNR (N, 1)
            z_data = f['Z'][:]
            
            # Convert one-hot to integer labels (0-23)
            self.labels = np.argmax(y_data, axis=1)  #Y:(N,)
            self.snrs = z_data.flatten()#Z：(N,)
            

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Lazy initialization of h5py.File object per process
        if self.archive is None:
            self.archive = h5py.File(self.file_path, 'r')


        # Read X data: (N, 1024, 2) -> (2, 1024)
        x = self.archive['X'][idx]  # Shape: (1024, 2)
        x = x.transpose(1, 0)             # Shape: (2, 1024)
        
        label = self.labels[idx]  # Integer label (0-23)
        z_snr = self.snrs[idx]    # SNR value (float)
        
        # Convert to torch tensors
        return torch.from_numpy(x).float(), torch.tensor(label, dtype=torch.long), torch.tensor(z_snr, dtype=torch.float)
