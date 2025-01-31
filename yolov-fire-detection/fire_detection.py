import cv2
from ultralytics import YOLO

# Eğitilmiş modeli yükle
model = YOLO('runs/detect/train/weights/best.pt')  # Eğitilen modelin ağırlık dosyasının yolu

# Kameradan görüntü yakalama
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # YOLOv8 ile yangın tespiti yapma
    results = model(frame)

    # Sonuçları görselleştirme
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, 'Fire', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow('Fire Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()