from ultralytics import YOLO
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="True"
# Load the pre-trained YOLOv8 model
model = YOLO('yolov8n.pt')
# Use the model for object detection
results = model('ch10_testimage.jpg')

# Print and visualize inspection results
for result in results:
    boxes = result.boxes  # Get the detected box
    print(boxes)  # Print the inspection results

    # Visualize inspection results
    result.plot()
    plt.imshow(result.plot())
    plt.savefig('detected_image.png')

#Save the detected box information directly to the file
with open('detections.txt', 'w') as f:
    for result in results:
        f.write(str(result.boxes))
