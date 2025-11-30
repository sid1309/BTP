import os

# --- Paths ---
images_dir = "/home/admin/Documents/BTP/dataset/image/test"
labels_dir = "/home/admin/Documents/BTP/dataset/labels/test"

# --- Supported image extensions ---
image_exts = {".jpg", ".jpeg", ".png"}

# --- Collect base filenames (without extensions) ---
image_basenames = {
    os.path.splitext(f)[0]
    for f in os.listdir(images_dir)
    if os.path.splitext(f)[1].lower() in image_exts
}

label_basenames = {
    os.path.splitext(f)[0]
    for f in os.listdir(labels_dir)
    if f.endswith(".txt")
}

# --- Find images missing labels ---
missing_labels = image_basenames - label_basenames

# --- Print summary ---
print(f"🖼️ Total images: {len(image_basenames)}")
print(f"🏷️ Total labels: {len(label_basenames)}")
print(f"⚠️ Images without labels: {len(missing_labels)}")
