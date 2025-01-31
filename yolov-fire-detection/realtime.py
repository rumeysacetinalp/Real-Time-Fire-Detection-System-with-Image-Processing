import cv2
import numpy as np
from ultralytics import YOLO
import math

# Model Yükleme
model_path = 'C:\\Users\\LENOVO\\yolov-fire-detection\\100best.pt'
model = YOLO(model_path)

# Video Kaynağı (0: Web Kamerası)
cap = cv2.VideoCapture(0)  # Canlı kamera akışı kullanımı
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output_video_path = 'C:\\Users\\LENOVO\\yolov-fire-detection\\output_video_.mp4'
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

# Fonksiyonlar
def draw_arrow(img, start, end, color=(255, 0, 0), thickness=2):
    start = tuple(map(int, start))
    end = tuple(map(int, end))
    cv2.arrowedLine(img, start, end, color, thickness)

def draw_box(img, box, label, score, color=(255, 0, 0), thickness=2):
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    cv2.putText(img, f'{label} {score:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

def calculate_centroid(box):
    x1, y1, x2, y2 = box
    return (int((x1 + x2) / 2), int((y1 + y2) / 2))

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

def calculate_area(box):
    """Calculates the area of a bounding box."""
    x1, y1, x2, y2 = box
    width = x2 - x1
    height = y2 - y1
    return width * height

# Başlangıç Değerleri
prev_smoke_center = None
prev_fire_center = None
prev_smoke_position = None
prev_fire_position = None
time_interval = 1 / fps

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # İnferans
    results = model.predict(source=frame)
    fire_boxes = []
    smoke_boxes = []
    for result in results:
        detections = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        labels = result.names
        for box, score, cls in zip(detections, scores, result.boxes.cls.cpu().numpy()):
            label = labels[int(cls)]
            if label == 'fire':
                fire_boxes.append((box, score))
            elif label == 'smoke':
                smoke_boxes.append((box, score))
    
    # Merkezlerin Hesaplanması
    smoke_center = None
    fire_center = None
    
    fire_centroids = [calculate_centroid(box) for box, _ in fire_boxes]
    smoke_centroids = [calculate_centroid(box) for box, _ in smoke_boxes]
    
    # Hız Hesaplama
    if smoke_boxes:
        smoke_center = smoke_centroids[0]
        if prev_smoke_position is not None:
            prev_center = calculate_centroid(prev_smoke_position)
            distance = np.sqrt((smoke_center[0] - prev_center[0]) ** 2 + (smoke_center[1] - prev_center[1]) ** 2)
            smoke_speed = distance / time_interval
        else:
            smoke_speed = 0
        prev_smoke_position = smoke_boxes[0][0]
    else:
        smoke_speed = 0
    
    if fire_boxes:
        fire_center = fire_centroids[0]
        if prev_fire_position is not None:
            prev_center = calculate_centroid(prev_fire_position)
            distance = np.sqrt((fire_center[0] - prev_center[0]) ** 2 + (fire_center[1] - prev_center[1]) ** 2)
            fire_speed = distance / time_interval
        else:
            fire_speed = 0
        prev_fire_position = fire_boxes[0][0]
    else:
        fire_speed = 0
    
    # Yön Hesaplama
    if fire_centroids and smoke_centroids and prev_fire_center and prev_smoke_center:
        avg_prev_center = (np.mean([prev_smoke_center[0], prev_fire_center[0]]), 
                           np.mean([prev_smoke_center[1], prev_fire_center[1]]))
        avg_current_center = (np.mean([smoke_center[0], fire_center[0]]), 
                              np.mean([smoke_center[1], fire_center[1]]))
        
        dx = avg_current_center[0] - avg_prev_center[0]
        dy = avg_current_center[1] - avg_prev_center[1]
        
        direction = calculate_direction(dx, dy)
    else:
        direction = "Unknown"

    prev_smoke_center = smoke_center
    prev_fire_center = fire_center

    # Yoğunluk Hesaplamaları
    smoke_area = sum([calculate_area(box[:4]) for box, _ in smoke_boxes])
    fire_area = sum([calculate_area(box[:4]) for box, _ in fire_boxes])
    total_area = smoke_area + fire_area

    frame_area = frame.shape[0] * frame.shape[1]  # Frame'in toplam alanı
    density = total_area / frame_area  # Yoğunluk hesaplama
    
    # Ekrana Yazma
    cv2.putText(frame, f"Smoke Speed: {smoke_speed:.2f}", (width - 400, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(frame, f"Fire Speed: {fire_speed:.2f}", (width - 400, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(frame, f"Direction: {direction}", (width - 400, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(frame, f"Density: {density:.2f}", (width - 400, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # Kutu ve Yön Çizimi
    for (box, score) in fire_boxes:
        draw_box(frame, box, 'fire', score, color=(0, 0, 255))
    for (box, score) in smoke_boxes:
        draw_box(frame, box, 'smoke', score, color=(0, 255, 0))
    if fire_centroids and smoke_centroids:
        fire_center = np.mean(fire_centroids, axis=0)
        smoke_center = np.mean(smoke_centroids, axis=0)
        draw_arrow(frame, fire_center, smoke_center)
    
    # Frame Yazma
    out.write(frame)
    cv2.imshow('Frame', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Kaynakları Serbest Bırakma
cap.release()
out.release()
cv2.destroyAllWindows()
print("Video processing complete. Output video saved at:", output_video_path)
