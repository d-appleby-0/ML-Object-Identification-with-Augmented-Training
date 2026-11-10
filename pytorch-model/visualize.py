import os
import torch
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from parse_annotations import load_btxrd_annotations
from test import get_object_detection_model

def visualize_inference(weights_path, sample_img_path, target_boxes, device, score_thresh=0.5):
    # 1. Initialize and load trained weights
    model = get_object_detection_model(num_classes=2)
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=device))
        print(f"Loaded weights from {weights_path}")
    else:
        print(f"Warning: {weights_path} not found. Running with initial pretrained weights.")
    
    model.to(device)
    model.eval()

    # 2. Run inference
    raw_img = Image.open(sample_img_path).convert("RGB")
    transform = transforms.ToTensor()
    img_tensor = transform(raw_img).unsqueeze(0).to(device)

    with torch.no_grad():
        predictions = model(img_tensor)[0]

    pred_boxes = predictions['boxes'][predictions['scores'] >= score_thresh].cpu().numpy()
    pred_scores = predictions['scores'][predictions['scores'] >= score_thresh].cpu().numpy()

    # 3. Draw Ground Truth
    gt_img = cv2.imread(sample_img_path)
    gt_img = cv2.cvtColor(gt_img, cv2.COLOR_BGR2RGB)
    for box in target_boxes:
        xmin, ymin, xmax, ymax = [int(v) for v in box]
        cv2.rectangle(gt_img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 3)

    # 4. Draw Predictions
    pred_img = cv2.imread(sample_img_path)
    pred_img = cv2.cvtColor(pred_img, cv2.COLOR_BGR2RGB)
    for box, score in zip(pred_boxes, pred_scores):
        xmin, ymin, xmax, ymax = [int(v) for v in box]
        cv2.rectangle(pred_img, (xmin, ymin), (xmax, ymax), (255, 0, 0), 3)
        cv2.putText(pred_img, f"{score:.2f}", (xmin, max(ymin - 10, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    # 5. Plot comparison
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(gt_img)
    axes[0].set_title("Ground Truth (Green)")
    axes[0].axis("off")

    axes[1].imshow(pred_img)
    axes[1].set_title(f"Predictions (Red, score >= {score_thresh})")
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    weights_file = "PyTorch-Model/outputs/best_fasterrcnn_btxrd.pth"
    annot_path = "PyTorch-Model/data/BTXRD/Annotations"
    image_dir = "PyTorch-Model/data/BTXRD/images"

    labels_dict = load_btxrd_annotations(annot_path)
    sample_img_name = next(iter(labels_dict.keys()))
    sample_path = os.path.join(image_dir, sample_img_name)
    sample_boxes = labels_dict[sample_img_name]['boxes']

    visualize_inference(weights_file, sample_path, sample_boxes, device)
