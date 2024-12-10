from ultralytics import YOLO
import matplotlib.pyplot as plt
import cv2
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

# 載入預訓練 YOLOv8 模型
model = YOLO('yolov8n.pt')

# 開啟影片檔案
video_path = 'video.mp4'
cap = cv2.VideoCapture(video_path)

# 初始化繪圖窗口
plt.ion()  # 開啟交互模式
fig, ax = plt.subplots(figsize=(8, 6))  # 設定視窗大小
manager = plt.get_current_fig_manager()
manager.window.wm_geometry("+100+100")  # 固定視窗位置 (X=100, Y=100)

# 初始化影像繪圖
im = ax.imshow([[0]], aspect='auto')  # 初始化空白影像
ax.axis('off')  # 移除座標軸

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 使用 YOLO 模型進行物件偵測
    result = model(frame)
    detections = result[0]

    # 計算人數 (假設 0 代表 "person")
    num_people = len([cls for cls in detections.boxes.cls if int(cls) == 0])
    print(f"The number of people in this frame: {num_people}")

    # 繪製偵測結果
    annotated_frame = detections.plot()  # 繪製標註過的影像
    im.set_data(annotated_frame)  # 更新影像
    plt.draw()  # 更新畫面
    plt.pause(0.0001)  # 暫停並更新畫面

cap.release()
plt.ioff()  # 關閉交互模式
plt.show()  # 防止視窗過早關閉
