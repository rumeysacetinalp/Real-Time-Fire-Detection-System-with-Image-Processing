import cv2
import numpy as np
from ultralytics import YOLO
import math

model = YOLO('C:\\Users\\LENOVO\\yolov-fire-detection\\best60.pt')

def detect_smoke_and_fire(frame):
    results = model(frame)
    smoke_boxes = []
    fire_boxes = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            confidence = box.conf[0]
            if int(box.cls) == 1: 
                smoke_boxes.append((x1, y1, x2, y2, confidence))
            elif int(box.cls) == 0:  
                fire_boxes.append((x1, y1, x2, y2, confidence))
    return smoke_boxes, fire_boxes

def calculate_direction(dx, dy):
    angle = math.degrees(math.atan2(dy, dx))
    if -22.5 <= angle < 22.5:
        return "East"
    elif 22.5 <= angle < 67.5:
        return "North-East"
    elif 67.5 <= angle < 112.5:
        return "North"
    elif 112.5 <= angle < 157.5:
        return "North-West"
    elif 157.5 <= angle < 180 or -180 <= angle < -157.5:
        return "West"
    elif -157.5 <= angle < -112.5:
        return "South-West"
    elif -112.5 <= angle < -67.5:
        return "South"
    elif -67.5 <= angle < -22.5:
        return "South-East"
    return "Unknown"

video_path = 'C:\\Users\\LENOVO\\yolov-fire-detection\\WhatsApp Video 2024-07-15 at 02.08.09.mp4'
cap = cv2.VideoCapture(video_path)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output_video.mp4', fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

ret, frame = cap.read()
prev_smoke_center = None
prev_fire_center = None
prev_smoke_position = None
prev_fire_position = None
time_interval = 1 / cap.get(cv2.CAP_PROP_FPS)  

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    smoke_boxes, fire_boxes = detect_smoke_and_fire(frame)  

    smoke_center = None
    fire_center = None

    if smoke_boxes:
        x1, y1, x2, y2, confidence = smoke_boxes[0]
        smoke_center = ((x1 + x2) // 2, (y1 + y2) // 2)
        
        if prev_smoke_position is not None:
            prev_center = (prev_smoke_position[0] + prev_smoke_position[2] // 2, prev_smoke_position[1] + prev_smoke_position[3] // 2)
            distance = np.sqrt((smoke_center[0] - prev_center[0]) ** 2 + (smoke_center[1] - prev_center[1]) ** 2)
            smoke_speed = distance / time_interval
        else:
            smoke_speed = 0

        prev_smoke_position = (x1, y1, x2, y2)

        # Smoke Bounding Box and Confidence
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2) 
        cv2.putText(frame, f"Smoke: {confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    else:
        smoke_speed = 0

    if fire_boxes:
        x1, y1, x2, y2, confidence = fire_boxes[0]
        fire_center = ((x1 + x2) // 2, (y1 + y2) // 2)
        
        if prev_fire_position is not None:
            prev_center = (prev_fire_position[0] + prev_fire_position[2] // 2, prev_fire_position[1] + prev_fire_position[3] // 2)
            distance = np.sqrt((fire_center[0] - prev_center[0]) ** 2 + (fire_center[1] - prev_center[1]) ** 2)
            fire_speed = distance / time_interval
        else:
            fire_speed = 0

        prev_fire_position = (x1, y1, x2, y2)

        # Fire Bounding Box and Confidence
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2) 
        cv2.putText(frame, f"Fire: {confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    else:
        fire_speed = 0

    # Direction Calculation
    if smoke_center and fire_center and prev_smoke_center and prev_fire_center:
        avg_prev_center = ((prev_smoke_center[0] + prev_fire_center[0]) // 2, 
                           (prev_smoke_center[1] + prev_fire_center[1]) // 2)
        avg_current_center = ((smoke_center[0] + fire_center[0]) // 2, 
                              (smoke_center[1] + fire_center[1]) // 2)
        
        dx = avg_current_center[0] - avg_prev_center[0]
        dy = avg_current_center[1] - avg_prev_center[1]

        direction = calculate_direction(dx, dy)
    else:
        direction = "Unknown"

    prev_smoke_center = smoke_center
    prev_fire_center = fire_center

    # Displaying Speed and Direction on Screen
    height, width, _ = frame.shape
    cv2.putText(frame, f"Smoke Speed: {smoke_speed:.2f} ", 
                (width - 400, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
    cv2.putText(frame, f"Fire Speed: {fire_speed:.2f} ", 
                (width - 400, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
    cv2.putText(frame, f"Direction: {direction}", 
                (width - 400, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

    out.write(frame)
    cv2.imshow('Frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
