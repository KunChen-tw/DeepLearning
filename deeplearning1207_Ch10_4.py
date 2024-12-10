from ultralytics import YOLO
import os
from PIL import Image

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

# Load the YOLOv8 model
model = YOLO('yolov8n.pt')  # You can choose different model files like 'yolov8s.pt', etc.

# Perform object detection
image_path = 'image2.jpg'
results = model(image_path)

# Get image dimensions for normalization
image = Image.open(image_path)
image_width, image_height = image.size

# Directory to save YOLO format results
output_dir = 'yolo_format_results'
os.makedirs(output_dir, exist_ok=True)

# Iterate through results
for idx, result in enumerate(results):
    boxes = result.boxes  # Detected boxes
    file_name = os.path.splitext(os.path.basename(image_path))[0] + f"_{idx}.txt"
    file_path = os.path.join(output_dir, file_name)

    with open(file_path, 'w') as f:
        for box in boxes:
            cls = int(box.cls.item())  # Class index
            conf = box.conf.item()  # Confidence score
            xmin, ymin, xmax, ymax = box.xyxy[0].tolist()  # Extract [xmin, ymin, xmax, ymax]

            # Normalize coordinates to YOLO format
            x_center = (xmin + xmax) / 2 / image_width
            y_center = (ymin + ymax) / 2 / image_height
            width = (xmax - xmin) / image_width
            height = (ymax - ymin) / image_height

            # Write to file in YOLO format
            f.write(f"{cls} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

print(f"Results saved in YOLO format under '{output_dir}' directory.")
