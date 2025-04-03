import os
import urllib.request
import zipfile
import shutil
from tqdm import tqdm

def download_and_extract_tiny_imagenet():
    """
    Downloads and extracts the Tiny-ImageNet dataset.
    """
    url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
    download_path = "tiny-imagenet-200.zip"
    extract_path = "."
    
    # Create a directory for the dataset if it doesn't exist
    if os.path.exists("tiny-imagenet-200"):
        print("Dataset directory already exists. Skipping download.")
        return
    
    print(f"Downloading Tiny-ImageNet from {url}...")
    
    # Define a custom progress bar for download
    class DownloadProgressBar(tqdm):
        def update_to(self, b=1, bsize=1, tsize=None):
            if tsize is not None:
                self.total = tsize
            self.update(b * bsize - self.n)
    
    # Download with progress bar
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc="Downloading") as t:
        urllib.request.urlretrieve(url, filename=download_path, reporthook=t.update_to)
    
    print("Download complete. Extracting files...")
    
    # Extract the zip file
    with zipfile.ZipFile(download_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    
    print("Extraction complete.")
    
    # Remove the zip file to save space
    os.remove(download_path)
    print("Zip file removed.")
    
    # Reorganize validation folder to match training folder structure
    organize_validation_folder()
    
    print("Dataset preparation complete.")

def organize_validation_folder():
    """
    Reorganizes the validation folder to match the structure of the training folder.
    This makes it easier to use standard PyTorch datasets.
    """
    val_dir = "tiny-imagenet-200/val"
    
    if not os.path.exists(os.path.join(val_dir, "images")):
        print("Validation folder already organized. Skipping.")
        return
    
    print("Reorganizing validation folder structure...")
    
    # Load validation annotations
    val_annotations_file = os.path.join(val_dir, "val_annotations.txt")
    val_annotations = {}
    
    with open(val_annotations_file, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            val_annotations[parts[0]] = parts[1]
    
    # Create class directories in validation folder
    for class_id in set(val_annotations.values()):
        os.makedirs(os.path.join(val_dir, class_id, "images"), exist_ok=True)
    
    # Move images to respective class directories
    for img_name, class_id in tqdm(val_annotations.items(), desc="Moving validation images"):
        src_path = os.path.join(val_dir, "images", img_name)
        dst_path = os.path.join(val_dir, class_id, "images", img_name)
        if os.path.exists(src_path):
            shutil.move(src_path, dst_path)
    
    # Clean up
    if os.path.exists(os.path.join(val_dir, "images")):
        shutil.rmtree(os.path.join(val_dir, "images"))
    
    print("Validation folder reorganization complete.")

if __name__ == "__main__":
    download_and_extract_tiny_imagenet()