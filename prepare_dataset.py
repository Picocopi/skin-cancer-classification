import os
import shutil
import pandas as pd

# Paths (based on your folder structure)
base_dir = "HAM10000"
metadata_file = os.path.join(base_dir, "HAM10000_metadata.csv")
image_folder = os.path.join("HAM10000_images")
output_folder = "ham10000_processed"  # This is the folder to use in train.py

# Read metadata
df = pd.read_csv(metadata_file)

# Create output directory
os.makedirs(output_folder, exist_ok=True)

# Process each image
for _, row in df.iterrows():
    img_file = row["image_id"] + ".jpg"
    label = row["dx"]  # e.g. mel, nv, bcc

    src_path = os.path.join(image_folder, img_file)
    dst_dir = os.path.join(output_folder, label)
    dst_path = os.path.join(dst_dir, img_file)

    # Create class folder if it doesn't exist
    os.makedirs(dst_dir, exist_ok=True)

    # Copy the image to its class folder
    if os.path.exists(src_path):
        shutil.copy(src_path, dst_path)
    else:
        print(f"⚠️ Missing image: {img_file}")

print("✅ All images sorted into folders in 'ham10000_processed/'")
