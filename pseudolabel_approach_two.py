from imports import *

# ----------------------
# CONFIGURATION
# ----------------------
MODEL_PATH = "baseline_model.pt"  # Path to your trained model weights
UNLABELED_IMAGES_DIR = "unlabeled/images"   # Path to folder with unlabeled images
OUTPUT_LABELS_DIR = "unlabeled/pseudo_labels"                   # Output directory for pseudo-labels
CONFIDENCE_THRESHOLD = 0.8  # Use a high threshold to ensure high-quality pseudo-labels

# Ensure output directory exists
os.makedirs(OUTPUT_LABELS_DIR, exist_ok=True)

# ----------------------
# LOAD TRAINED MODEL
# ----------------------
model = YOLO(MODEL_PATH)  # Loads your YOLOv8 model

# ----------------------
# PROCESS UNLABELED IMAGES
# ----------------------
# Supports common image extensions; adjust if needed
image_extensions = ("*.jpg", "*.jpeg", "*.png")
image_paths = []
for ext in image_extensions:
    image_paths.extend(glob.glob(os.path.join(UNLABELED_IMAGES_DIR, ext)))

print(f"Found {len(image_paths)} images in {UNLABELED_IMAGES_DIR}")

for img_path in image_paths:
    # Read image using OpenCV
    img = cv2.imread(img_path)
    if img is None:
        print(f"Could not load image {img_path}. Skipping.")
        continue

    # Run inference on the image
    results = model(img, conf=CONFIDENCE_THRESHOLD)

    # Prepare a list to store pseudo-label lines (YOLO format)
    pseudo_label_lines = []

    # Loop through each result (usually one per image)
    for result in results:
        # Loop through each detected box in the result
        for box in result.boxes:
            # Although the model call applies the confidence threshold, we double-check here
            if box.conf[0] < CONFIDENCE_THRESHOLD:
                continue

            # Extract normalized coordinates: (x_center, y_center, width, height)
            x_center, y_center, width, height = box.xywhn[0].tolist()
            # Get the predicted class id (assuming classes are indexed 0 and 1)
            class_id = int(box.cls[0])
            # Format the label line and add to the list
            pseudo_label_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

    # Extract image name (without extension) to use for label file name
    img_name = os.path.splitext(os.path.basename(img_path))[0]
    label_file_path = os.path.join(OUTPUT_LABELS_DIR, f"{img_name}.txt")

    if pseudo_label_lines:
        # If detections were found, write the pseudo-labels to the file
        with open(label_file_path, "w") as label_file:
            for line in pseudo_label_lines:
                label_file.write(line + "\n")
        print(f"Saved pseudo-labels to {label_file_path}")
    else:
        # If no detections were found, ensure no label file remains
        if os.path.exists(label_file_path):
            os.remove(label_file_path)
        print(f"No detections found for {img_path}. No pseudo-label file created.")

        # Optionally, if you have copied the image to a target folder and want to remove it:
        os.remove(img_path)  # Uncomment this line if you want to delete the image

print("✅ Pseudo-labeling complete!")

source_folder = 'unlabeled/images'
destination_folder = 'w1609/images/train'

# Ensure the destination folder exists
os.makedirs(destination_folder, exist_ok=True)

# Iterate over all files in the source folder
for filename in os.listdir(source_folder):
    # Construct full file path
    source_file = os.path.join(source_folder, filename)
    destination_file = os.path.join(destination_folder, filename)
    
    # Only copy if it is a file (not a directory)
    if os.path.isfile(source_file):
        shutil.copy(source_file, destination_file)
        print(f"Copied {source_file} to {destination_file}")


source_folder = 'unlabeled/pseudo_labels'
destination_folder = 'w1609/labels/train'

# Ensure the destination folder exists
os.makedirs(destination_folder, exist_ok=True)

# Iterate over all files in the source folder
for filename in os.listdir(source_folder):
    # Construct full file path
    source_file = os.path.join(source_folder, filename)
    destination_file = os.path.join(destination_folder, filename)
    
    # Only copy if it is a file (not a directory)
    if os.path.isfile(source_file):
        shutil.copy(source_file, destination_file)
        print(f"Copied {source_file} to {destination_file}")


LABELED_DATA_DIR = "w1609"  # Update with your path  # Update with your path
NUM_CLASSES = 2  # weed vs. crop
BATCH_SIZE = 5
EPOCHS = 50
WARMUP_EPOCHS = 2  # Train only on labeled data first

dataset2_yaml = f"""
path: C:/Users/khara/OneDrive/Documents/Kriti2
train: {LABELED_DATA_DIR}/images/train  # Path to labeled images
val: {LABELED_DATA_DIR}/images/test   # Path to validation images (can be the same as train if no validation set)
nc: 2  # Number of classes
names: ['weed', 'crop']  # Class names
"""

with open("dataset2.yaml", "w") as f:
    f.write(dataset2_yaml)

MODEL_PATH = "runs/detect/weed_detection/weights/best.pt"
model = YOLO(MODEL_PATH)

result=model.train(
    data="dataset2.yaml",  # Path to your custom dataset YAML file
    epochs=EPOCHS,
    batch=BATCH_SIZE,
    imgsz=512,
    device="cuda" if torch.cuda.is_available() else "cpu",  # Use GPU if available
    name="weed_detection"
)
