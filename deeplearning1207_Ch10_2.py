from ultralytics import YOLO
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="True"
# Load the pre-trained YOLOv8 model
model = YOLO('yolov8n.pt') 
# Use the model for object detection
results = model('ch10_testimage2.jpg')

# Print and visualize inspection results
for result in results:
    boxes = result.boxes  # Get the detected box

    # Visualize inspection results
    result.plot()
    
    #To save the visualization to a file, you can use matplotlib
    plt.imshow(result.plot())
    plt.savefig('detected_image2.png')
    
    print("The number of people: ", len(boxes.cls))
