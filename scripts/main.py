# Core PyTorch and PyTorch Lightning
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import albumentations as A
from albumentations.pytorch import ToTensorV2
from prepare_data import GeoImageDataModule , GeoImageDataset
from models import GeoImageSegmentModel
import cv2
from utils import post_process_mask

def train_model(
    image_dir,
    mask_dir,
    checkpoint_dir,
    batch_size=4,
    num_workers=2,
    lr=1e-4,
    weight_decay=1e-3,
    encoder_name="resnet34",
    max_epochs=100,
    patience=10,
    precision="16-mixed",
    augs_train=None,
    augs_val=None,
    resume_from_checkpoint=None
):
    """
    A complete training function that encapsulates the entire training workflow.

    Args:
        image_dir (str): Path to the directory with training/validation images.
        mask_dir (str): Path to the directory with training/validation masks.
        checkpoint_dir (str): Path to the directory where checkpoints will be saved.
        batch_size (int, optional): Batch size for training. Defaults to 4.
        num_workers (int, optional): Number of workers for the dataloader. Defaults to 2.
        lr (float, optional): Learning rate. Defaults to 1e-4.
        weight_decay (float, optional): Weight decay for the optimizer. Defaults to 1e-3.
        encoder_name (str, optional): Encoder to use for the model. Defaults to "resnet34".
        max_epochs (int, optional): Maximum number of epochs to train for. Defaults to 100.
        patience (int, optional): Patience for early stopping. Defaults to 10.
        precision (int or str, optional): Training precision (e.g., 16 or '16-mixed'). Defaults to 16.
        augs_train (A.Compose, optional): Custom training augmentations. If None, uses a strong default.
        augs_val (A.Compose, optional): Custom validation augmentations. If None, uses a default.
        resume_from_checkpoint (str, optional): Path to a checkpoint to resume training from. Defaults to None.
    """
    # Define default augmentations if none are provided
    if augs_train is None:
        augs_train = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.3),
            A.Affine(
                scale=(0.9, 1.1),
                translate_percent=0.05,
                rotate=(-15, 15),
                p=0.5,
                border_mode=cv2.BORDER_CONSTANT
            ),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
            A.GaussianBlur(blur_limit=(3, 7), p=0.1),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])

    if augs_val is None:
        augs_val = A.Compose([
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])

    # Initialize DataModule
    data_module = GeoImageDataModule(
        image_dir=image_dir,
        mask_dir=mask_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        augmentations_train=augs_train,
        augmentations_val=augs_val
    )

    # Initialize Model
    model = GeoImageSegmentModel(lr=lr, weight_decay=weight_decay, encoder_name=encoder_name)

    # Initialize Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename=f"{encoder_name}-best-model-{{epoch:02d}}-{{val_iou:.4f}}",
        save_top_k=1,
        monitor="val_iou",
        mode="max"
    )
    early_stopping_callback = EarlyStopping(monitor="val_iou", patience=patience, mode="max")

    # Initialize Trainer
    trainer = pl.Trainer(
        accelerator="auto",
        devices=1,
        precision=precision,
        max_epochs=max_epochs,
        callbacks=[checkpoint_callback, early_stopping_callback]
    )

    # Start Training
    trainer.fit(
        model,
        datamodule=data_module,
        ckpt_path=resume_from_checkpoint
    )
    
if __name__ == "__main__":
    # Example usage
    IMAGE_DIR = "data/AOI_3_Paris/PS-RGB"
    MASK_DIR = "data/AOI_3_Paris/geojson_roads"
    CHECKPOINT_DIR = "checkpoints"
    
    train_model(
        image_dir=IMAGE_DIR,
        mask_dir=MASK_DIR,
        checkpoint_dir=CHECKPOINT_DIR,
    )