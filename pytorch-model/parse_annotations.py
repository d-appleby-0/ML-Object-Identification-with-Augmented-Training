import os
import json

def load_btxrd_annotations(annotations_dir):
    """
    Parses BTXRD Labelme-style JSON files in annotations_dir.
    Maps image filenames (e.g. 'IMG000001.jpeg') to bounding boxes [xmin, ymin, xmax, ymax].
    """
    labels_dict = {}
    
    if not os.path.exists(annotations_dir):
        print(f"Warning: Directory not found: {annotations_dir}")
        return labels_dict

    for filename in os.listdir(annotations_dir):
        if not filename.endswith('.json'):
            continue
            
        json_path = os.path.join(annotations_dir, filename)
        # Match JSON filename to image filename (e.g., IMG000001.json -> IMG000001.jpeg)
        img_name = filename.replace('.json', '.jpeg')
        
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        boxes = []
        labels = []
        
        for shape in data.get("shapes", []):
            points = shape.get("points", [])
            
            # Format 1: Rectangle with 2 diagonal points [[xmin, ymin], [xmax, ymax]]
            if len(points) == 2:
                x1, y1 = points[0]
                x2, y2 = points[1]
                xmin = min(x1, x2)
                ymin = min(y1, y2)
                xmax = max(x1, x2)
                ymax = max(y1, y2)
                boxes.append([float(xmin), float(ymin), float(xmax), float(ymax)])
                labels.append(1)  # Class 1: Tumor / Lesion
                
            # Format 2: Polygon shape with multiple vertex points
            elif len(points) > 2:
                xs = [p[0] for p in points]
                ys = [p[1] for p in points]
                boxes.append([float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys))])
                labels.append(1)

        if boxes:
            labels_dict[img_name] = {
                'boxes': boxes,
                'labels': labels
            }
            
    return labels_dict
