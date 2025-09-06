import torch
import numpy as np
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchmetrics.classification import BinaryJaccardIndex
import cv2
import matplotlib.pyplot as plt
from models import GeoImageSegmentModel
from prepare_data import GeoImageDataModule, GeoImageDataset
from utils import post_process_mask, visualize_image


def run_evaluation(
    use_close,
    close_kernel_size,
    use_min_object_size,
    min_object_size,
    image_dir,
    mask_dir,
    checkpoint_path,
    max_batches=None
):
    """
    Loads a trained model from a checkpoint, runs it over a validation/test set,
    and computes the IoU for both the raw and post-processed predictions.

    Args:
        use_close (bool): If True, applies morphological closing.
        close_kernel_size (int): The size of the kernel for morphological closing.
        use_min_object_size (bool): If True, removes small objects from the thick mask.
        min_object_size (int): The minimum pixel area for an object to be kept.
        image_dir (str): Path to the directory of evaluation images.
        mask_dir (str): Path to the directory of evaluation masks.
        checkpoint_path (str): Path to the .ckpt file of the trained model.
        batch_size (int, optional): Batch size for evaluation. Defaults to 8.
        num_workers (int, optional): Number of dataloader workers. Defaults to 2.
        max_batches (int): limits the number of batches to evaluate. 
    """
    # --- 1. Load Model ---
    model = GeoImageSegmentModel.load_from_checkpoint(checkpoint_path=CHECKPOINT_PATH)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Model loaded and moved to {device}")

    # --- 2. Prepare Data ---
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
    
    augs_val = A.Compose([
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
    data_module = GeoImageDataModule(
    image_dir=image_dir,
    mask_dir=mask_dir,
    batch_size=4,
    num_workers=2,
    augmentations_train=augs_train,
    augmentations_val=augs_val
    )
    data_module.setup()
    val_dataloader = data_module.val_dataloader()

    # --- 3. Run Evaluation Loop ---
    raw_iou_metric = BinaryJaccardIndex().to(device)
    post_processed_iou_metric = BinaryJaccardIndex().to(device)

    print("\nStarting evaluation over the entire validation set...")
    for batch_idx, batch in enumerate(tqdm(val_dataloader, desc="Evaluating")):
        if max_batches is not None and batch_idx >= max_batches:
            break

        images, ground_truth_masks = batch
        images = images.to(device)
        ground_truth_masks = ground_truth_masks.unsqueeze(1).to(device)

        with torch.no_grad():
            logits = model(images)
            probabilities = torch.sigmoid(logits)
            raw_predictions = (probabilities > 0.5)

        # Update the raw IoU metric
        raw_iou_metric.update(raw_predictions, ground_truth_masks.int())

        # Post-process the predictions and update the second metric
        post_processed_preds_batch = []
        for i in range(images.shape[0]):
            raw_pred_np = raw_predictions[i].squeeze().cpu().numpy()
            post_processed_pred = post_process_mask(
                raw_pred_np,
                use_close=use_close,
                close_kernel_size=close_kernel_size,
                use_min_object_size=use_min_object_size,
                min_object_size=min_object_size,
            )
            post_processed_preds_batch.append(post_processed_pred)

        # Convert the list of post-processed numpy arrays to a tensor for the metric
        pp_preds_tensor = torch.from_numpy(np.stack(post_processed_preds_batch)).unsqueeze(1).to(device)
        post_processed_iou_metric.update(pp_preds_tensor, ground_truth_masks.int())

    # --- 4. Print Final Results ---
    # Compute the final macro-IoU score from the accumulated stats
    avg_raw_iou = raw_iou_metric.compute()
    avg_pp_iou = post_processed_iou_metric.compute()

    print("\n--- Evaluation Complete ---")
    print(f"Average RAW Prediction IoU:      {avg_raw_iou.item():.4f}")
    print(f"Average POST-PROCESSED IoU:      {avg_pp_iou.item():.4f}")

    return avg_pp_iou

def plot_evaluation_results(
    CHECKPOINT_PATH,
    image_dir,
    mask_dir,
):
    """
    Loads a trained model from a checkpoint, runs it over a batch from the validation set,
    and visualizes the original images, ground truth masks, raw predictions, and post-processed predictions.

    Args:
        CHECKPOINT_PATH (str): Path to the .ckpt file of the trained model.
        image_dir (str): Path to the directory of evaluation images.
        mask_dir (str): Path to the directory of evaluation masks.
    """
    # --- 1. Load Your Trained Model ---
    model = GeoImageSegmentModel.load_from_checkpoint(checkpoint_path=CHECKPOINT_PATH)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Model loaded and moved to {device}")


    # --- 2. Set up the DataModule to get a batch ---
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
        A.RandomBrightnessContrast(  
            brightness_limit=0.2,
            contrast_limit=0.2,
            p=0.3
        ),
        A.GaussianBlur(blur_limit=(3, 7), p=0.1),  
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),   
        ToTensorV2()
    ])

    augs_val = A.Compose([
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    data_module = GeoImageDataModule(
        image_dir=image_dir,
        mask_dir=mask_dir,
        batch_size=4,
        num_workers=2,
        augmentations_train=augs_train,
        augmentations_val=augs_val
    )
    data_module.setup()

    # Create the dataloader
    val_dataloader = data_module.val_dataloader()

    # Create the iterator object ONCE
    data_iterator = iter(val_dataloader)

    # Get the FIRST batch
    images, ground_truth_masks = next(data_iterator)
    print("Fetched batch 1")

    # --- 3. Get Model Predictions for the Batch ---
    images = images.to(device)
    with torch.no_grad():
        logits = model(images)
        probabilities = torch.sigmoid(logits)
        predictions = (probabilities > 0.5).cpu()


    # --- 4. Visualize Results for Each Image in the Batch ---

    batch_size = images.shape[0]
    plt.figure(figsize=(20, 5 * batch_size)) 

    for i in range(batch_size):
        # Get the i-th image, ground truth, and prediction from the batch

        image_to_plot = images[i]
        gt_mask_to_plot = ground_truth_masks[i].squeeze()
        pred_mask_to_plot = predictions[i].squeeze()

        # --- Apply the full post-processing pipeline ---
        final_prediction = post_process_mask(pred_mask_to_plot.numpy())

        # --- Plotting ---
        # Column 1: Original Image
        plt.subplot(batch_size, 4, i * 4 + 1)
        plt.imshow(visualize_image(image_to_plot))
        plt.title(f"Image {i+1}")
        plt.axis("off")

        # Column 2: Ground Truth
        plt.subplot(batch_size, 4, i * 4 + 2)
        plt.imshow(gt_mask_to_plot, cmap='gray')
        plt.title(f"Ground Truth {i+1}")
        plt.axis("off")

        # Column 3: Raw Model Prediction (Thick)
        plt.subplot(batch_size, 4, i * 4 + 3)
        plt.imshow(pred_mask_to_plot, cmap='gray')
        plt.title(f"Raw Prediction {i+1}")
        plt.axis("off")

        # Column 4: Fully Post-Processed Prediction
        plt.subplot(batch_size, 4, i * 4 + 4)
        plt.imshow(final_prediction, cmap='gray')
        plt.title(f"Post-Processed {i+1}")
        plt.axis("off")

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    # This is an example of how you would run the evaluation from a script
    
    # Define the paths to your data and best model
    CHECKPOINT_PATH = "checkpoints/baseline-best-model-epoch=76-val_iou=0.5967.ckpt"
    IMAGE_DIR = "data/AOI_3_Paris/PS-RGB"
    MASK_DIR = "data/AOI_3_Paris/geojson_roads"
    BATCH_SIZE = 6 # Use a larger batch size for evaluating more of the dataset (the val dataset will have 13 batches in total)

    # Define the post-processing parameters you want to test
    # These could be the best ones you found from your Optuna study
    best_params = {
        'use_close': True,
        'close_kernel_size': 5,
        'use_min_object_size': True,
        'min_object_size': 1500,
    }

    # Run the evaluation
    run_evaluation(use_close=True,
                        close_kernel_size=5
                        ,use_min_object_size=True,
                        min_object_size=1500,
                        max_batches=BATCH_SIZE,
                        image_dir=IMAGE_DIR,
                        mask_dir=MASK_DIR,
                        checkpoint_path=CHECKPOINT_PATH)
    
    # Visualize the results (the first batch)
    plot_evaluation_results(CHECKPOINT_PATH=CHECKPOINT_PATH,
                            image_dir=IMAGE_DIR,
                            mask_dir=MASK_DIR)
