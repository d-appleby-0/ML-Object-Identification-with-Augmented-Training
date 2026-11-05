PyTorch Bone Tumor Identification Model Training

NOTES

DATA PROCESSING 
  - Return tuples to ensure security.
  - Images should be passed in either as a 3D Tensor [3, H, W] or an array with height and width dimensions.
    - In a 3D tensor, the first index represents layers of image processed through different colors (Three represents once in R G B)
  
  
OBJECT DETECTION
  - Utilize TorchVision Library

  - Instance Segmentation Tools (Pivoted to Object Detection - but ideally this would be best assuming unlimited RAM)
    - Generate segmentation masks utilizing built in Mask R-CNN in companion library (factory functions).
    - Segmentation masks improve accuracy dramatically with tradeoff in speed.

  - Utilize built-in object detection functions via PyTorch due to immense tradeoff in speed seen by Mask R-CNN
    - torchvision.models.detection module
    - Utilize Faster R-CNN or SSDLITE (object detection functions)

    - Faster R-CNN functions are built on various backbones which have differing speed, accuracy, and memory usage.
    - Use default backbone (resnet50_fpn) prioritizing accuracy over immense speed.


SKELETON

  Medical Image Digestion
  - Use pydicom to parse .dcm files to read pixels
  - Normalize window levels to standard tensor float range... [0.0, 1.0]
  - Duplicate single normalized slice to satisfy 3D tensor structure
  - Establish bounding box coordinates (x min, y min, x max, y max)

  Detection Model
  - RPN scans high-probability regions
  - Fast R-CNN extracts texture matrices and outputs bounding box coordinates ALONGSIDE probabilities. (pre-built loss functions included)
  - Use resnet50_fpn backbone prioritizing accuracy over speed when benchmarking.
  - Ensure geometric invariance via random flips

  Testing / Training
  - Measure mAP and recall

