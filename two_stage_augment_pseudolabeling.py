from imports import *


# Define the weak augmentation pipeline
weak_augmentations = A.Compose([
    A.HorizontalFlip(p=0.5),          
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=10, p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),    
    A.GaussianBlur(blur_limit=(3, 3), p=0.3),
    A.CoarseDropout(max_holes=1, max_height=10, max_width=10, p=0.2)
])

# Define input and output directories
input_dir = "unlabeled/images"  # Directory containing input images
output_dir = "unlabeled/weak_agumentation"  # Directory to save augmented images
os.makedirs(output_dir, exist_ok=True)

# Loop through all images in the input directory
for filename in os.listdir(input_dir):
    if filename.endswith(('.jpg', '.png', '.jpeg')):  # Filter for image files
        image_path = os.path.join(input_dir, filename)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        
        # Generate multiple augmented versions for each image
        for i in range(2):  # Adjust the range for more/less augmentations
            augmented = weak_augmentations(image=image)['image']
            augmented_filename = f"{os.path.splitext(filename)[0]}_aug_{i+1}.jpg"
            save_path = os.path.join(output_dir, augmented_filename)
            cv2.imwrite(save_path, cv2.cvtColor(augmented, cv2.COLOR_RGB2BGR))  # Convert RGB back to BGR
            print(f"Saved: {save_path}")


# ----------------------
# CONFIGURATION
# ----------------------
MODEL_PATH = "baseline_model.pt"  # Path to your trained model weights
UNLABELED_IMAGES_DIR = "unlabeled/weak_agumentation"   # Path to folder with unlabeled images
OUTPUT_LABELS_DIR = "unlabeled/pseudo_labels"                   # Output directory for pseudo-labels
CONFIDENCE_THRESHOLD = 0.85  # Use a high threshold to ensure high-quality pseudo-labels

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

source_folder = 'unlabeled/weak_agumentation'
destination_folder = 'labeled/images'

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
destination_folder = 'labeled/labels'

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


NUM_CLASSES = 2  # weed vs. crop
BATCH_SIZE = 5
EPOCHS = 40
WARMUP_EPOCHS = 2  # Train only on labeled data first

dataset_yaml = f"""
train: labeled/images  # Path to labeled images
val: test/images    # Path to validation images (can be the same as train if no validation set)
nc: 2  # Number of classes
names: ['weed', 'crop']  # Class names
"""

with open("/kaggle/working/dataset.yaml", "w") as f:
    f.write(dataset_yaml)

result=model.train(
    data="dataset.yaml",  # Path to your custom dataset YAML file
    epochs=EPOCHS,
    batch=BATCH_SIZE,
    imgsz=512,
    device="cuda" if torch.cuda.is_available() else "cpu",  # Use GPU if available
    name="weed_detection_weak_0.75"
)

# Define the strong augmentation pipeline
strong_augmentations = A.Compose([
    A.RandomResizedCrop(height=256, width=256, scale=(0.6, 1.0), p=0.5),  # Random crop and resize
    A.HorizontalFlip(p=0.5),  # Flip horizontally
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=30, p=0.8),  # Large translation, scaling, rotation
    A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.7),  # Strong brightness/contrast
    A.GaussianBlur(blur_limit=(5, 7), p=0.5),  # Stronger blur
    A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),  # Add random noise
    A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.4),  # Elastic distortions
    A.GridDistortion(p=0.3),  # Grid distortions
])

# Define input and output directories
input_dir = "unlabeled/images"  # Directory containing input images
output_dir = "unlabeled/strong_agumentation"  # Directory to save augmented images
os.makedirs(output_dir, exist_ok=True)

# Loop through all images in the input directory
for filename in os.listdir(input_dir):
    if filename.endswith(('.jpg', '.png', '.jpeg')):  # Filter for image files
        image_path = os.path.join(input_dir, filename)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        
        # Generate multiple augmented versions for each image
        for i in range(2):  # Adjust the range for more/less augmentations
            augmented = strong_augmentations(image=image)['image']
            augmented_filename = f"{os.path.splitext(filename)[0]}_aug_{i+1}.jpg"
            save_path = os.path.join(output_dir, augmented_filename)
            cv2.imwrite(save_path, cv2.cvtColor(augmented, cv2.COLOR_RGB2BGR))  # Convert RGB back to BGR
            print(f"Saved: {save_path}")

# ----------------------
# CONFIGURATION
# ----------------------
MODEL_PATH = "best (1).pt"  # Path to your trained model weights
UNLABELED_IMAGES_DIR = "unlabeled/strong_agumentation"   # Path to folder with unlabeled images
OUTPUT_LABELS_DIR = "unlabeled/pseudo_labels"                   # Output directory for pseudo-labels
CONFIDENCE_THRESHOLD = 0.85  # Use a high threshold to ensure high-quality pseudo-labels

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

source_folder = 'unlabeled/strong_agumentation'
destination_folder = 'labeled/images'

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
destination_folder = 'labeled/labels'

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

MODEL_PATH = "best (1).pt"
model = YOLO(MODEL_PATH)

NUM_CLASSES = 2  # weed vs. crop
BATCH_SIZE = 5
EPOCHS = 40
WARMUP_EPOCHS = 2  # Train only on labeled data first

dataset_yaml = f"""
train: labeled/images  # Path to labeled images
val: test/images    # Path to validation images (can be the same as train if no validation set)
nc: 2  # Number of classes
names: ['weed', 'crop']  # Class names
"""

with open("dataset.yaml", "w") as f:
    f.write(dataset_yaml)

result=model.train(
    data="dataset.yaml",  # Path to your custom dataset YAML file
    epochs=EPOCHS,
    batch=BATCH_SIZE,
    imgsz=512,
    device="cuda" if torch.cuda.is_available() else "cpu",  # Use GPU if available
    name="weed_detection_weak_0.75"
)