from torch.utils.data import Dataset
import torchio as tio
import os
import glob

# Preprocessing transforms (normalization and resizing)
# Adjust target_shape if your images have different dimensions (e.g., more depth slices)
PREPROCESSING_TRANSFORMS = tio.Compose([
    tio.RescaleIntensity(out_min_max=(-1, 1)),
    tio.CropOrPad(target_shape=(256, 256, 32))
])

# Data augmentation transforms for training
TRAIN_TRANSFORMS = tio.Compose([
    tio.RandomFlip(axes=(1), flip_probability=0.5),
])

class FOMO60kDataset(Dataset):
    def __init__(self, root_dir: str, imgtype: str = 'flair', augmentation: bool = False):
        super().__init__()
        self.root_dir = root_dir
        self.imgtype = imgtype
        self.augmentation = augmentation
        
        # Define transforms
        self.preprocessing = PREPROCESSING_TRANSFORMS
        self.transforms = TRAIN_TRANSFORMS if augmentation else None
        
        # Load file list
        self.file_paths = self.get_data_files()
        print(f"FOMO60k Dataset loaded: {len(self.file_paths)} images found.")

    def get_data_files(self):
        # Search for files matching the pattern: root/sub_*/ses_*/<imgtype>.nii.gz
        # Example: root/sub_2473/ses_1/flair.nii.gz
        pattern = os.path.join(self.root_dir, 'sub_*', 'ses_*', f'{self.imgtype}.nii.gz')
        file_paths = glob.glob(pattern)
        
        # Fallback: try searching for uncompressed .nii files if .nii.gz not found
        if not file_paths:
            print("No .nii.gz files found. Checking for .nii files...")
            pattern = os.path.join(self.root_dir, 'sub_*', 'ses_*', f'{self.imgtype}.nii')
            file_paths = glob.glob(pattern)
            
        return sorted(file_paths)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx: int):
        path = self.file_paths[idx]
        
        # Load image using Torchio
        img = tio.ScalarImage(path)
        
        # Apply preprocessing (Normalize/Crop)
        img = self.preprocessing(img)
        
        # Apply augmentation (only if enabled)
        if self.transforms:
            img = self.transforms(img)
            
        # Return data in (C, D, H, W) format
        # Permute needed to align with model expectations (depth as 2nd dim)
        return {'data': img.data.permute(0, -1, 1, 2)}