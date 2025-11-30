import xml.etree.ElementTree as ET
import os
import argparse
from tqdm import tqdm # Optional: for a progress bar (pip install tqdm)

# --- Configuration ---
# 1. DEFINE YOUR CLASS MAPPING: Map XML class names to YOLO integer IDs (starting from 0)
#    - Check your XML files for exact class names used (case-sensitive).
#    - Only include classes you want your final model to detect.
class_mapping = {
    # --- Verify these names match your XML <name> tags ---
    "car": 0,
    "autorickshaw": 1,
    "truck": 2,
    "motorcycle": 3,
    "bus": 4,
    "person": 5,
    # Add/Remove classes as needed. Ensure IDs are 0 to N-1.
}

# 2. SET BASE PATHS: Adjust these paths to match your directory structure
#    Example: If your IDD_Detection folder is in /home/admin/datasets/
#    idd_base_dir = '/home/admin/datasets/IDD_Detection'
#    output_base_dir = '/home/admin/datasets/IDD_Detection_YOLO_Labels' # Or same as idd_base_dir
idd_base_dir = '/home/admin/Documents/BTP/Dataset/IDD_Detection' # Contains Annotations, JPEGImages, train.txt etc.
output_base_dir = '/home/admin/Documents/BTP/Dataset/IDD_Detection/YOLO_Labels' # Directory WHERE the 'labels' folder will be created

# --- End Configuration ---

def convert_bbox_to_yolo(size, box):
    """Converts Pascal VOC bbox [xmin, ymin, xmax, ymax] to YOLO format [cx, cy, w, h] normalized."""
    dw = 1.0 / size[0] # image width
    dh = 1.0 / size[1] # image height
    x = (box[0] + box[1]) / 2.0 # center x (xmin + xmax) / 2
    y = (box[2] + box[3]) / 2.0 # center y (ymin + ymax) / 2
    w = box[1] - box[0] # width (xmax - xmin)
    h = box[3] - box[2] # height (ymax - ymin)

    norm_x = x * dw
    norm_y = y * dh
    norm_w = w * dw
    norm_h = h * dh

    # Clamp values to [0.0, 1.0] to prevent out-of-bounds errors
    norm_x = max(0.0, min(1.0, norm_x))
    norm_y = max(0.0, min(1.0, norm_y))
    norm_w = max(0.0, min(1.0, norm_w))
    norm_h = max(0.0, min(1.0, norm_h))

    return (norm_x, norm_y, norm_w, norm_h)

def process_split_file(split_file_path, idd_root_dir, output_root_dir):
    """Processes annotations listed in a split file (train.txt, val.txt, test.txt)."""

    annotations_base_dir = os.path.join(idd_root_dir, 'Annotations')
    # Determine the split name (train, val, test) from the file name
    split_name = os.path.splitext(os.path.basename(split_file_path))[0]
    output_labels_dir = os.path.join(output_root_dir, 'labels', split_name)

    processed_files = 0
    skipped_files = 0
    missing_xml_count = 0

    if not os.path.exists(split_file_path):
        print(f"Error: Split file not found at {split_file_path}")
        return

    print(f"Reading image/annotation list from: {split_file_path}")
    try:
        with open(split_file_path, 'r') as f:
            # Read lines and strip any leading/trailing whitespace
            relative_paths = [line.strip() for line in f if line.strip()]
    except Exception as e:
        print(f"Error reading split file {split_file_path}: {e}")
        return

    if not relative_paths:
        print(f"Warning: No valid paths found in {split_file_path}. Is the file empty or paths formatted incorrectly?")
        return

    print(f"Processing {len(relative_paths)} entries for the '{split_name}' split...")
    print(f"Annotations source base: {annotations_base_dir}")
    print(f"Output labels base:      {output_labels_dir}")

    for relative_path in tqdm(relative_paths, desc=f"Converting '{split_name}' annotations"):
        # Construct full path to XML file based on relative path from split file
        # Assumes paths in txt are like 'folder/subfolder/basename' without extension
        xml_filename = relative_path + '.xml'
        xml_path = os.path.join(annotations_base_dir, xml_filename)

        # Construct full path for output TXT file, mirroring structure
        output_txt_filename = relative_path + '.txt'
        output_txt_path = os.path.join(output_labels_dir, output_txt_filename)

        # Create output subdirectories if they don't exist
        try:
            os.makedirs(os.path.dirname(output_txt_path), exist_ok=True)
        except OSError as e:
            print(f"Error creating directory {os.path.dirname(output_txt_path)}: {e}. Skipping file.")
            skipped_files += 1
            continue

        if not os.path.exists(xml_path):
            # Keep track but don't print for every missing file unless debugging
            missing_xml_count += 1
            skipped_files += 1
            continue # Skip to the next file in the list

        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()

            # Find image size from XML
            size_element = root.find('size')
            if size_element is None:
                 # print(f"Warning: Cannot find <size> tag in {xml_filename}. Skipping.")
                 skipped_files += 1
                 continue

            img_width_elem = size_element.find('width')
            img_height_elem = size_element.find('height')
            if img_width_elem is None or img_height_elem is None or not img_width_elem.text or not img_height_elem.text:
                # print(f"Warning: Cannot find valid <width> or <height> in {xml_filename}. Skipping.")
                skipped_files += 1
                continue

            img_width = int(img_width_elem.text)
            img_height = int(img_height_elem.text)
            if img_width <= 0 or img_height <= 0:
                # print(f"Warning: Invalid image dimensions ({img_width}x{img_height}) in {xml_filename}. Skipping.")
                skipped_files += 1
                continue
            size = (img_width, img_height)

            yolo_annotations = []
            for obj in root.findall('object'):
                class_name_elem = obj.find('name')
                if class_name_elem is None or not class_name_elem.text: continue
                class_name = class_name_elem.text.strip() # Use strip() for safety

                if class_name in class_mapping:
                    class_id = class_mapping[class_name]
                    bndbox_elem = obj.find('bndbox')
                    if bndbox_elem is None: continue

                    # Extract coordinates safely
                    try:
                        xmin = float(bndbox_elem.find('xmin').text)
                        ymin = float(bndbox_elem.find('ymin').text)
                        xmax = float(bndbox_elem.find('xmax').text)
                        ymax = float(bndbox_elem.find('ymax').text)
                    except (AttributeError, ValueError, TypeError): # Handle missing tags or non-numeric text
                        # print(f"Warning: Invalid bbox coordinate in {xml_filename} for {class_name}. Skipping object.")
                        continue

                    # Basic sanity check for coordinates
                    if xmin >= xmax or ymin >= ymax:
                        # print(f"Warning: Invalid bbox values (min >= max) in {xml_filename} for {class_name}. Skipping object.")
                        continue

                    # Pass coordinates in order: xmin, xmax, ymin, ymax for calculation
                    box = (xmin, xmax, ymin, ymax)
                    bb_yolo = convert_bbox_to_yolo(size, box)
                    yolo_annotations.append(f"{class_id} {bb_yolo[0]:.6f} {bb_yolo[1]:.6f} {bb_yolo[2]:.6f} {bb_yolo[3]:.6f}")

            # Write annotations to file (even if the list is empty for an image)
            with open(output_txt_path, 'w') as out_f:
                 if yolo_annotations:
                      out_f.write("\n".join(yolo_annotations) + "\n")
                 # else: pass # Creates an empty file if no mapped objects found

            processed_files += 1

        except ET.ParseError:
            # print(f"Error: Could not parse XML {xml_filename}. Skipping.")
            skipped_files += 1
        except Exception as e:
            # print(f"An unexpected error occurred processing {xml_filename}: {e}")
            skipped_files += 1

    print(f"\nConversion Complete for '{split_name}' split.")
    print(f"Processed: {processed_files} files.")
    print(f"Skipped:   {skipped_files} files (due to errors or missing XML).")
    if missing_xml_count > 0:
         print(f"Info: {missing_xml_count} XML files listed in {split_file_path} were not found in {annotations_base_dir}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert IDD XML annotations (listed in a split file) to YOLO format.")
    parser.add_argument('split_file', type=str, help="Path to the split file (e.g., train.txt, val.txt, test.txt)")
    args = parser.parse_args()

    # --- IMPORTANT ---
    # 1. EDIT class_mapping above. Ensure names EXACTLY match XML <name> tags.
    # 2. EDIT idd_base_dir and output_base_dir paths above.
    # --- -------------

    # Basic check if base directories exist
    if not os.path.isdir(idd_base_dir):
        print(f"Error: IDD base directory not found: {idd_base_dir}")
        exit(1)
    if not os.path.isdir(os.path.join(idd_base_dir, 'Annotations')):
        print(f"Error: Annotations folder not found inside: {idd_base_dir}")
        exit(1)

    # Create the base output directory if it doesn't exist
    os.makedirs(output_base_dir, exist_ok=True)

    process_split_file(args.split_file, idd_base_dir, output_base_dir)

    print(f"\nReminder: Run this script again for other split files (e.g., val.txt, test.txt)")
    print(f"Output labels are being saved under: {os.path.join(output_base_dir, 'labels')}")
