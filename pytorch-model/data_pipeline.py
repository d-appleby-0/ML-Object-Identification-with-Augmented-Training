import os
import pydicom
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image, ImageDraw
import torchvision

class DicomDataset(Dataset):
    """Custom PyTorch Dataset for loading medical images."""
    
    def __init__(self, dicom_dir, labels_dict=None, transforms=None):
        self.dicom_dir = dicom_dir
        self.labels_dict = labels_dict or {}
        
        # If labels_dict is passed, use ONLY its keys (caps dataset to split size)
        if self.labels_dict:
            self.img_names = list(self.labels_dict.keys())
        else:
            valid_exts = ('.dcm', '.jpeg', '.jpg', '.png')
            self.img_names = sorted([f for f in os.listdir(dicom_dir) if f.lower().endswith(valid_exts)])
            
        self.transforms = transforms

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_path = os.path.join(self.dicom_dir, img_name)

        image = Image.open(img_path).convert("RGB")

        # Prepare Object Detection Targets
        target = {}

        if img_name in self.labels_dict and len(self.labels_dict[img_name]['boxes']) > 0:
            boxes = torch.as_tensor(self.labels_dict[img_name]['boxes'], dtype=torch.float32)
            labels = torch.as_tensor(self.labels_dict[img_name]['labels'], dtype=torch.int64)

            area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            iscrowd = torch.zeros((len(boxes),), dtype=torch.int64)
        else: 
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            area = torch.zeros((0,), dtype=torch.float32)
            iscrowd = torch.zeros((0,), dtype=torch.int64)

        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = torch.tensor([idx])
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            image = self.transforms(image)

        # Return img_name as well if save_predicted_boxes needs it
        return image, target

    @staticmethod
    def save_predicted_boxes(model, dataset, device, output_dir="outputs/saved_predictions", score_thresh=0.5):
        """Runs inference on dataset samples and saves overlayed bounding boxes to disk."""  
        os.makedirs(output_dir, exist_ok=True)
        model.eval()

        print(f"\n[+] Saving bounding box prediction images to '{output_dir}'...")

        with torch.no_grad():
            for i in range(min(20, len(dataset))):
                image_tensor, target = dataset[i]
                img_name = dataset.img_names[i]
                
                # Predict bounding boxes
                predictions = model([image_tensor.to(device)])[0]
                
                # Reconstruct image tensor back to PIL for visual drawing
                orig_img = torchvision.transforms.functional.to_pil_image(image_tensor)
                draw = ImageDraw.Draw(orig_img)

                boxes = predictions['boxes'].cpu().numpy()
                scores = predictions['scores'].cpu().numpy()
                labels = predictions['labels'].cpu().numpy()

                for box, score, label in zip(boxes, scores, labels):
                    if score >= score_thresh:
                        xmin, ymin, xmax, ymax = box
                        # Draw red bounding rectangle
                        draw.rectangle([xmin, ymin, xmax, ymax], outline="red", width=3)
                        draw.text((xmin, max(0, ymin - 10)), f"Tumor: {score:.2f}", fill="red")

                # Save annotated image file to disk
                save_path = os.path.join(output_dir, f"pred_{img_name}")
                orig_img.save(save_path)

        print(f"Saved prediction images successfully.")
