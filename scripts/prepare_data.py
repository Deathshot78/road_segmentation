import os
import re
import numpy as np
import cv2
import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader, random_split
import rasterio
from rasterio.features import rasterize
import geopandas as gpd
import albumentations as A
from albumentations.pytorch import ToTensorV2
from projects.road_segmentation.scripts.utils import run_command

def download_and_prepare_data(output_dir="data"):
    """
    Downloads and extracts the SpaceNet 3 dataset for Paris (Train and Test).
    
    Args:
        output_dir (str): The directory to download and extract the data into.
    """
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    print(f"Data will be downloaded and extracted to: '{output_dir}'")

    # List of files to download
    s3_files_to_download = [
        "s3://spacenet-dataset/spacenet/SN3_roads/tarballs/SN3_roads_train_AOI_3_Paris.tar.gz",
        "s3://spacenet-dataset/spacenet/SN3_roads/tarballs/SN3_roads_train_AOI_3_Paris_geojson_roads_speed.tar.gz",
        "s3://spacenet-dataset/spacenet/SN3_roads/tarballs/SN3_roads_test_public_AOI_3_Paris.tar.gz"
    ]

    tar_files_to_extract = [
        "SN3_roads_train_AOI_3_Paris.tar.gz",
        "SN3_roads_train_AOI_3_Paris_geojson_roads_speed.tar.gz",
        "SN3_roads_test_public_AOI_3_Paris.tar.gz"
    ]

    # --- 1. Download Files ---
    for s3_path in s3_files_to_download:
        filename = os.path.basename(s3_path)
        local_path = os.path.join(output_dir, filename)
        
        if os.path.exists(local_path):
            print(f"File '{filename}' already exists. Skipping download.")
        else:
            command = ["aws", "s3", "cp", s3_path, local_path, "--no-sign-request"]
            run_command(command)

    # --- 2. Extract Files ---
    # The final directory that will be created
    final_data_dir = os.path.join(output_dir, "AOI_3_Paris")
    
    # Check if the data has already been extracted
    if os.path.exists(final_data_dir):
         print(f"Directory '{final_data_dir}' already exists. Assuming data is extracted. Skipping extraction.")
    else:
        for filename in tar_files_to_extract:
            local_path = os.path.join(output_dir, filename)
            command = ["tar", "-xzvf", local_path]
            # We run the command from within the output directory
            run_command(command, working_dir=output_dir)

    print("\nData preparation complete!")
    print(f"Your data should be located in: '{final_data_dir}'")

class GeoImageDataset(Dataset):
    """
    PyTorch Dataset for loading and preprocessing satellite images and their
    corresponding road network masks from GeoJSON files.

    This class handles:
    - Pairing of image (.tif) and mask (.geojson) files based on filenames.
    - Robustly processing geospatial data to ensure perfect pixel alignment
      between images and rasterized masks, regardless of their original
      Coordinate Reference Systems (CRS).
    - Applying data augmentations using the Albumentations library.
    - Padding images and masks to be compatible with model architectures that
      require specific input dimensions (e.g., divisible by 16 or 32).
    """
    def __init__(self, image_dir, mask_dir, augmentations=None):
        """
        Args:
            image_dir (str): Path to the directory containing image files (.tif).
            mask_dir (str): Path to the directory containing mask files (.geojson).
            augmentations (albumentations.Compose, optional): An Albumentations
                pipeline to apply to the image and mask. Defaults to None.
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.augmentations = augmentations

        if not os.path.isdir(image_dir): raise FileNotFoundError(f"Image directory not found: {image_dir}")
        if not os.path.isdir(mask_dir): raise FileNotFoundError(f"Mask directory not found: {mask_dir}")

        self.image_files = sorted([f for f in os.listdir(image_dir) if f.endswith((".tif", ".tiff"))])
        self.mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith(".geojson")])

        self.index_to_mask = {self.extract_index(f): f for f in self.mask_files}

        self.paired_files = [
            (img, self.index_to_mask.get(self.extract_index(img)))
            for img in self.image_files
            if self.extract_index(img) in self.index_to_mask
        ]
        self.paired_files = [(img, mask) for img, mask in self.paired_files if mask is not None]
        if not self.paired_files:
            print(f"CRITICAL WARNING: No image-mask pairs found after attempting to match filenames in {image_dir}.")

    def extract_index(self, filename):
        """Extracts the numerical index from a filename (e.g., 'img123')."""
        base_name = os.path.splitext(filename)[0]
        match = re.search(r'img(\d+)', base_name)
        return match.group(1) if match else None

    def process_image(self, image_path):
        """
        Loads a satellite image, performs contrast stretching, and returns it as a
        NumPy array along with its geographic metadata.
        
        Args:
            image_path (str): The path to the image file.

        Returns:
            tuple: A tuple containing:
                - np.ndarray: The processed image as a uint8 NumPy array (H, W, C).
                - dict: A dictionary of geographic information (crs, transform, etc.).
        """
        with rasterio.open(image_path) as src:
            geo_info = {"crs": src.crs, "transform": src.transform, "height": src.height, "width": src.width}
            image_data = src.read(list(range(1, min(src.count, 3) + 1))).astype(np.float32)
            if image_data.shape[0] < 3:
                padding = np.zeros((3 - image_data.shape[0], geo_info["height"], geo_info["width"]), dtype=np.float32)
                image_data = np.concatenate([image_data, padding], axis=0)

        stretched_bands = []
        for i in range(image_data.shape[0]):
             min_val, max_val = np.percentile(image_data[i], (2, 98))
             if np.isclose(max_val, min_val): stretched = np.zeros_like(image_data[i])
             else: stretched = (image_data[i] - min_val) / (max_val - min_val)
             stretched_bands.append(np.clip(stretched, 0, 1))
        stretched_image_np = np.stack(stretched_bands)
        t_img = stretched_image_np.transpose(1, 2, 0)
        img_uint8 = (t_img * 255).astype(np.uint8)
        return img_uint8, geo_info

    def rasterize_mask(self, mask_path, image_geo_info):
        """
        Creates a binary raster mask from a GeoJSON file, ensuring it is
        perfectly aligned with its corresponding source image. Uses a robust
        "Project-Buffer-Reproject" workflow.

        Args:
            mask_path (str): The path to the GeoJSON mask file.
            image_geo_info (dict): The geographic metadata from the source image.

        Returns:
            np.ndarray: The rasterized binary mask as a uint8 NumPy array (H, W).
        """
        try:
            gdf = gpd.read_file(mask_path)
        except Exception:
            return np.zeros((image_geo_info["height"], image_geo_info["width"]), dtype=np.uint8)

        if gdf.crs and gdf.crs != image_geo_info["crs"]:
            try: gdf = gdf.to_crs(image_geo_info["crs"])
            except Exception: return np.zeros((image_geo_info["height"], image_geo_info["width"]), dtype=np.uint8)

        if gdf.crs.is_geographic:
            try: gdf = gdf.to_crs("EPSG:32631")
            except Exception: return np.zeros((image_geo_info["height"], image_geo_info["width"]), dtype=np.uint8)

        valid_gdf = gdf[gdf.geometry.is_valid & ~gdf.geometry.is_empty]
        if valid_gdf.empty: return np.zeros((image_geo_info["height"], image_geo_info["width"]), dtype=np.uint8)

        # Using the buffer size that previously gave good results
        buffer_distance_meters = 3.0
        buffered_geometries = valid_gdf.geometry.buffer(buffer_distance_meters)
        buffered_gdf = gpd.GeoDataFrame(geometry=buffered_geometries, crs=gdf.crs)
        gdf_reprojected_for_raster = buffered_gdf.to_crs(image_geo_info["crs"])

        mask_array = rasterize(
            gdf_reprojected_for_raster.geometry,
            out_shape=(image_geo_info["height"], image_geo_info["width"]),
            transform=image_geo_info["transform"], fill=0, dtype='uint8', all_touched=True
        )
        if mask_array.ndim == 3: mask_array = mask_array.squeeze(0)
        return mask_array

    def __len__(self): return len(self.paired_files)

    def __getitem__(self, idx):
        image_name, mask_name = self.paired_files[idx]
        image_path = os.path.join(self.image_dir, image_name)
        image_np, geo_info = self.process_image(image_path)
        mask_path = os.path.join(self.mask_dir, mask_name)
        mask_np = self.rasterize_mask(mask_path, geo_info)

        pad_transform = A.PadIfNeeded(min_height=1312, min_width=1312, border_mode=cv2.BORDER_CONSTANT)
        padded = pad_transform(image=image_np, mask=mask_np)
        image_np_padded, mask_np_padded = padded['image'], padded['mask']

        if self.augmentations:
            augmented = self.augmentations(image=image_np_padded, mask=mask_np_padded)
            image_tensor, mask_tensor = augmented['image'], augmented['mask']
        else:
            image_tensor = torch.from_numpy(image_np_padded.transpose(2,0,1)).float()
            mask_tensor = torch.from_numpy(mask_np_padded).unsqueeze(0).float()

        return image_tensor.float(), mask_tensor.float()

class GeoImageDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for the road segmentation task.
    Handles the creation of training, validation, and test dataloaders.
    """
    def __init__(self, image_dir, mask_dir, augmentations_train=None, augmentations_val=None, batch_size=4, num_workers=0, split_perc=0.8, **kwargs):
        super().__init__()
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.augmentations_train = augmentations_train
        self.augmentations_val = augmentations_val
        self.split_perc = split_perc

    def setup(self, stage=None):
        temp_dataset = GeoImageDataset(self.image_dir, self.mask_dir)
        num_paired_files = len(temp_dataset)
        train_size = int(self.split_perc * num_paired_files)
        val_size = num_paired_files - train_size

        g = torch.Generator().manual_seed(42)
        indices = torch.randperm(num_paired_files, generator=g).tolist()
        train_indices, val_indices = indices[:train_size], indices[train_size:]

        train_dataset_full = GeoImageDataset(self.image_dir, self.mask_dir, augmentations=self.augmentations_train)
        self.train_dataset = torch.utils.data.Subset(train_dataset_full, train_indices)

        val_dataset_full = GeoImageDataset(self.image_dir, self.mask_dir, augmentations=self.augmentations_val)
        self.val_dataset = torch.utils.data.Subset(val_dataset_full, val_indices)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True, persistent_workers=True if self.num_workers > 0 else False, pin_memory=True, drop_last=True)
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, persistent_workers=True if self.num_workers > 0 else False, pin_memory=True, drop_last=True)

# This block allows you to run the script directly from the command line
if __name__ == "__main__":
    download_and_prepare_data()
