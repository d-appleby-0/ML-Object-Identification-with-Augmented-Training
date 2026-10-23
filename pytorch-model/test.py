import data_pipeline as dp

dicom_dir = "./data"

dataset = dp.DicomDataset(dicom_dir=dicom_dir)

print("--- ENVIRONMENT CHECK ---")
print(f"Total DICOM files found: {len(dataset)}")

if len(dataset) > 0:
    # 3. Pull the real image (This triggers your float32 & normalization math)
    image, target = dataset[0]

    print("\n--- IMAGING PIPELINE TEST ---")
    print("Processed Image Shape:", image.shape)
    print("Image Data Type:      ", image.dtype)
    print("Target Metadata:      ", target)
