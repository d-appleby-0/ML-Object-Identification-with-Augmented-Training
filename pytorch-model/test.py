import data_pipeline as dp

dicom_dir = "./data"

dataset = dp.DicomDataset(dicom_dir=dicom_dir)

print(f"Total DICOM files found: {len(dataset)}")

if len(dataset) > 0:
    image, target = dataset[0]
    print("Processed Image Shape:", image.shape)
    print("Image Data Type:      ", image.dtype)
    print("Target Metadata:      ", target)
