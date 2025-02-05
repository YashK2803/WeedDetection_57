from imports import *


## Creating Yaml file

data_yaml_content = """
path: C:/Users/khara/OneDrive/Documents/Kriti2
train: labeled/images
val: test/images
nc: 2
names: ["weed", "crop"]
"""

# Save the YAML file
yaml_file_path = "data.yaml"
with open(yaml_file_path, "w") as file:
    file.write(data_yaml_content)

print(f"YAML file saved at: {yaml_file_path}")

# ----------------------
# CONFIGURATION
# ----------------------
MODEL_PATH = "baseline_model.pt"  # Path to your trained model weights
UNLABELED_IMAGES_DIR = "unlabeled/images"   # Path to folder with unlabeled images
TRAINING_IMAGES_DIR = "labeled/images"  # Path to folder with original training images
TRAINING_LABELS_DIR = "labeled/labels"  # Path to folder with original training labels
VAL_IMAGES_DIR="test/images"
OUTPUT_LABELS_DIR = "unlabeled/pseudo_labels"  # Output directory for pseudo-labels
CONFIDENCE_THRESHOLD = 0.85  # Confidence threshold for high-quality pseudo-labels
BATCH_SIZE = 100  # Process this many images per iteration
EPOCHS = 25  # Number of iterations (pseudo-training cycles)

# Ensure necessary directories exist
os.makedirs(OUTPUT_LABELS_DIR, exist_ok=True)
os.makedirs(TRAINING_IMAGES_DIR, exist_ok=True)
os.makedirs(TRAINING_LABELS_DIR, exist_ok=True)

# ----------------------
# LOAD TRAINED MODEL
# ----------------------
model = YOLO(MODEL_PATH)  # Load the YOLOv8 model

# ----------------------
# TRAIN ON BATCHES ITERATIVELY
# ----------------------
for epoch in range(EPOCHS):
    if epoch>=5 and epoch%5==0:
        CONFIDENCE_THRESHOLD -=0.05
    print(f"\nStarting epoch {epoch + 1}/{EPOCHS}")
    
    # Collect all remaining unlabeled images
    image_extensions = ("*.jpg", "*.jpeg", "*.png")
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(UNLABELED_IMAGES_DIR, ext)))

    if not image_paths:
        print("No more unlabeled images. Stopping training.")
        break

    # Take the first BATCH_SIZE images
    current_batch = image_paths[:BATCH_SIZE]
    print(f"Processing {len(current_batch)} images.")

    for img_path in current_batch:
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
                # Check if the confidence is above the threshold
                if box.conf[0] < CONFIDENCE_THRESHOLD:
                    continue

                # Extract normalized coordinates: (x_center, y_center, width, height)
                x_center, y_center, width, height = box.xywhn[0].tolist()
                # Get the predicted class id
                class_id = int(box.cls[0])
                # Format the label line and add to the list
                pseudo_label_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

        # Extract image name (without extension) to use for label file name
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        label_file_path = os.path.join(TRAINING_LABELS_DIR, f"{img_name}.txt")

        if pseudo_label_lines:
            # Save pseudo-labels if detections were found
            with open(label_file_path, "w") as label_file:
                for line in pseudo_label_lines:
                    label_file.write(line + "\n")

            # Move image to the training directory
            target_image_path = os.path.join(TRAINING_IMAGES_DIR, os.path.basename(img_path))
            os.rename(img_path, target_image_path)  # Move the file
            print(f"Added {img_path} to training set with pseudo-labels.")
        else:
            print(f"No high-confidence detections for {img_path}. Skipping.")

    # Train the model on the updated training set
    print("Training YOLO model on updated training set...")
    model.train(data=yaml_file_path, epochs=20, imgsz=512)


print("✅ Pseudo-labeling and training complete!")

