from ultralytics import YOLO
import matplotlib.pyplot as plt
import cv2
import os
from collections import Counter

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

# Load the pre-trained YOLOv8 model
model = YOLO('yolov8n.pt')

# Use the model for object detection
results = model('ch10_testimage3.jpg')

# Counter to store object class statistics
class_counter = Counter()

# Iterate through results
for result in results:
    boxes = result.boxes  # Get detected boxes

    # Count occurrences of each class
    for box in boxes:
        class_id = int(box.cls)
        class_counter[class_id] += 1

    # Use YOLO's result.plot() to draw all bounding boxes
    annotated_image = result.plot()  # Returns a numpy.ndarray in BGR format

    # Convert BGR to RGB for Matplotlib display
    # annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)

    # Save the annotated image using matplotlib
    plt.imshow(annotated_image)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig("detected_objects.png")
    plt.show()

# Print class statistics
print("Object counts:")
for class_id, count in class_counter.items():
    print(f"{model.names[class_id]}: {count}")
