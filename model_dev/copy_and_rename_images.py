import os
import shutil

# ==== CONFIGURATION ====
base_folder = "/home/admin/Documents/BTP/Dataset/IDD_Detection/JPEGImages"
train_txt = "/home/admin/Documents/BTP/Dataset/IDD_Detection/val.txt"
dest_folder = "/home/admin/Documents/BTP/dataset/image/val"

# Make sure destination exists
os.makedirs(dest_folder, exist_ok=True)

# Supported extensions
extensions = [".jpg", ".jpeg", ".png"]

# Read all lines from train.txt
with open(train_txt, "r") as f:
    lines = [line.strip() for line in f if line.strip()]

for rel_path in lines:
    # Build full path without extension
    full_base = os.path.join(base_folder, rel_path)


    # Try each extension
    found = False
    for ext in extensions:
        src_path = full_base + ext
        if os.path.exists(src_path):
            # Extract folder parts
            parts = rel_path.split("/")
            # Make new filename as folder1_folder2_..._filename.ext
            new_name = "_".join(parts) + ext
            dest_path = os.path.join(dest_folder, new_name)

            # Copy file
            shutil.copy(src_path, dest_path)
            print(f"Copied: {src_path} -> {dest_path}")
            found = True
            break

    if not found:
        print(f"⚠️ File not found for: {rel_path}")

print("✅ All done!")
