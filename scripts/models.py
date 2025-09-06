import torch
import torch.nn as nn
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.losses import DiceLoss
from torchmetrics.classification import BinaryJaccardIndex

class GeoImageSegmentModel(pl.LightningModule):
    """
    A PyTorch Lightning module for road segmentation from satellite images.

    This class encapsulates the model architecture (U-Net), the loss function (a
    combination of Dice and BCE), the optimization logic (AdamW with a cosine
    annealing scheduler), and the training/validation steps.
    """
    def __init__(self, lr=1e-4, weight_decay=1e-4, encoder_name="resnet50"):
        """
        Initializes the model, loss functions, and metrics.

        Args:
            lr (float, optional): The learning rate for the optimizer. Defaults to 1e-4.
            weight_decay (float, optional): The weight decay for the AdamW optimizer. Defaults to 1e-4.
            encoder_name (str, optional): The name of the encoder backbone to use from
                segmentation-models-pytorch. Defaults to "resnet50".
        """
        super().__init__()
        # Save hyperparameters to the checkpoint, allowing for easy reloading
        self.save_hyperparameters()

        # Initialize the U-Net model from the segmentation-models-pytorch library
        self.model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights='imagenet', # Use pre-trained ImageNet weights for transfer learning
            in_channels=3,
            classes=1 # Binary output: road or not road
        )

        # Define the two components of the combination loss function
        self.dice_loss = DiceLoss(mode='binary', from_logits=True)
        self.bce_loss = nn.BCEWithLogitsLoss()
        
        # Initialize the IoU metric from torchmetrics for validation
        self.iou_metric = BinaryJaccardIndex()

    def forward(self, x):
        """
        Performs a forward pass through the model.

        Args:
            x (torch.Tensor): The input batch of images.

        Returns:
            torch.Tensor: The raw output logits from the model.
        """
        return self.model(x)

    def _common_step(self, batch, batch_idx, stage):
        """
        A common function for the training and validation steps to avoid code duplication.

        Args:
            batch (tuple): A tuple containing the images and masks.
            batch_idx (int): The index of the current batch.
            stage (str): The current stage, either "train" or "val".

        Returns:
            torch.Tensor: The calculated loss for the batch.
        """
        images, masks = batch
        if masks.ndim == 3: masks = masks.unsqueeze(1)

        # Get raw model outputs (logits)
        outputs = self(images)
        
        # Calculate the combination loss, giving more weight to Dice loss
        # to better handle class imbalance.
        loss = 0.8 * self.dice_loss(outputs, masks) + 0.2 * self.bce_loss(outputs, masks.float())

        # Convert logits to probabilities and then to a binary prediction mask
        preds_prob = torch.sigmoid(outputs)
        preds_binary = (preds_prob > 0.5)
        
        # Calculate the IoU metric
        iou = self.iou_metric(preds_binary, masks.int())

        # Log the loss and IoU for monitoring in TensorBoard or other loggers
        self.log(f'{stage}_loss', loss, on_epoch=True, prog_bar=True, logger=True, batch_size=images.size(0))
        self.log(f'{stage}_iou', iou, on_epoch=True, prog_bar=True, logger=True, batch_size=images.size(0))

        return loss

    def training_step(self, batch, batch_idx):
        """Performs a single training step."""
        return self._common_step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        """Performs a single validation step."""
        return self._common_step(batch, batch_idx, "val")

    def configure_optimizers(self):
        """
        Configures the optimizer (AdamW) and learning rate scheduler (CosineAnnealingLR).

        Returns:
            dict: A dictionary containing the optimizer and the LR scheduler configuration.
        """
        optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=self.hparams.lr, 
            weight_decay=self.hparams.weight_decay
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=self.trainer.max_epochs, 
            eta_min=1e-6
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch", # Step the scheduler at the end of each epoch
            },
        }