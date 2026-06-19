from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")

image_path = "dataset/sample.jpg"

results = model(image_path)

image = cv2.imread(image_path)

person_count = 0

for result in results:
    for box in result.boxes:
        class_id = int(box.cls[0])

        if class_id == 0:
            person_count += 1

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            cv2.rectangle(
                image,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            cv2.putText(
                image,
                "Person",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )

cv2.imwrite("outputs/result.jpg", image)

print(f"People detected: {person_count}")
print("Output saved to outputs/result.jpg")