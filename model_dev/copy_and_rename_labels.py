import os
import shutil

# --- SOURCE AND DESTINATION PATHS ---
src_root = "Dataset/IDD_Detection/labels/val"
dst_root = "/home/admin/Documents/BTP/dataset/label/val"

os.makedirs(dst_root, exist_ok=True)

# Walk through all subfolders inside the train directory
for root, dirs, files in os.walk(src_root):
    path_parts = root.split(os.sep)
    
    if len(path_parts) >= 2:
        main_folder = path_parts[-2]   # e.g. frontFar
        sub_folder = path_parts[-1]    # e.g. BLR-2018-03-22_17-39-26_2_frontFar
    else:
        continue

    for file in files:
        if file.endswith(".txt"):
            src_path = os.path.join(root, file)
            new_filename = f"{main_folder}_{sub_folder}_{file}"
            dst_path = os.path.join(dst_root, new_filename)
            
            shutil.copy2(src_path, dst_path)
            print(f"Copied: {src_path} -> {dst_path}")

print("✅ All label files copied and renamed successfully!")
