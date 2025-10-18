import zipfile
import os

# Path to your zip file in /content after downloading from Google Drive
zip_path = "/content/drive/MyDrive/finaldata.csv.zip"   # change this to your actual zip filename
extract_dir = "/content"
target_file = "finaldata.csv"

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    # Check if the file exists inside the archive
    if target_file in zip_ref.namelist():
        # Extract only the target file
        zip_ref.extract(target_file, extract_dir)
        print(f"✅ Extracted {target_file} to {extract_dir}")
    else:
        print(f"⚠️ {target_file} not found in the zip file.")
