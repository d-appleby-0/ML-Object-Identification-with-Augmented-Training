import os
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from sklearn.model_selection import train_test_split

from data_pipeline import DicomDataset
from parse_annotations import load_btxrd_annotations

def collate_fn(batch):
    return tuple(zip(*batch))

def get_object_detection_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

def train_one_epoch(model, optimizer, data_loader, device):
    model.train()
    total_loss = 0.0

    for step, (images, targets) in enumerate(data_loader):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        total_loss += losses.item()

        if (step + 1) % 10 == 0 or (step + 1) == len(data_loader):
            print(f"  [Train] Batch [{step + 1}/{len(data_loader)}] - Loss: {losses.item():.4f}")

    return total_loss / len(data_loader)

@torch.no_grad()
def evaluate_val_loss(model, data_loader, device):
    """Computes validation loss by temporarily setting model to train mode to get loss dict."""
    model.train()
    total_loss = 0.0

    for images, targets in data_loader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        total_loss += losses.item()

    return total_loss / len(data_loader)

def main():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Using device: {device}")

    NUM_CLASSES = 2  # 0: Background, 1: Tumor
    BATCH_SIZE = 2
    NUM_EPOCHS = 2
    OUTPUT_DIR = "tested_weights"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    PRED_DIR = "outputs/saved_predictions"
    os.makedirs(PRED_DIR, exist_ok=True)

    transform_pipeline = transforms.Compose([
        transforms.ToTensor()
    ])

    annot_path = "data/BTXRD/Annotations"
    image_path = "data/BTXRD/images"

    # 1. Parse Annotations
    labels_dict = load_btxrd_annotations(annot_path)
    annotated_img_names = list(labels_dict.keys())
    print(f"Loaded annotations for {len(annotated_img_names)} images.")

    annotated_img_names = annotated_img_names[:200]

    # 2. Split annotated images 80/20 (Train / Val)
    train_names, val_names = train_test_split(annotated_img_names, test_size=0.2, random_state=42)
    train_labels = {k: labels_dict[k] for k in train_names}
    val_labels = {k: labels_dict[k] for k in val_names}

    print(f"Training split: {len(train_labels)} images | Validation split: {len(val_labels)} images")

    # 3. Create Datasets and DataLoaders
    train_dataset = DicomDataset(dicom_dir=image_path, labels_dict=train_labels, transforms=transform_pipeline)
    val_dataset = DicomDataset(dicom_dir=image_path, labels_dict=val_labels, transforms=transform_pipeline)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    # 4. Initialize Model & Optimizer
    model = get_object_detection_model(num_classes=NUM_CLASSES)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

    best_val_loss = float('inf')

    # 5. Training Loop with Checkpointing
    print("\nStarting Training Loop...")
    for epoch in range(NUM_EPOCHS):
        print(f"\n--- Epoch [{epoch + 1}/{NUM_EPOCHS}] ---")
        train_loss = train_one_epoch(model, optimizer, train_loader, device)
        val_loss = evaluate_val_loss(model, val_loader, device)

        print(f"Epoch [{epoch + 1}/{NUM_EPOCHS}] Summary - Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        # Save latest checkpoint
        latest_path = os.path.join(OUTPUT_DIR, "latest_fasterrcnn_btxrd.pth")
        torch.save(model.state_dict(), latest_path)

        # Save best model checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = os.path.join(OUTPUT_DIR, "best_fasterrcnn_btxrd.pth")
            torch.save(model.state_dict(), best_path)
            print(f"  --> Saved new best model checkpoint to {best_path} (Val Loss: {val_loss:.4f})")

        epoch_output_dir = os.path.join(PRED_DIR, f"saved_predictions/epoch_{epoch + 1}")
        DicomDataset.save_predicted_boxes(
            model=model, 
            dataset=val_dataset, 
            device=device, 
            output_dir=epoch_output_dir, 
            score_thresh=0.15  # Lower threshold so early epoch detections show up
        )
if __name__ == "__main__":
    main()
