from ultralytics import YOLO
import cv2
from datetime import datetime

model = YOLO("yolov8n.pt")

image_path = "dataset/sample.jpg"

results = model(image_path)

image = cv2.imread(image_path)

person_count = 0
confidence_sum = 0

for result in results:
    for box in result.boxes:

        class_id = int(box.cls[0])

        if class_id == 0:

            person_count += 1

            confidence = float(box.conf[0])
            confidence_sum += confidence

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            label = f"Person {confidence:.2f}"

            cv2.rectangle(
                image,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            cv2.putText(
                image,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )

cv2.imwrite("outputs/result.jpg", image)

if person_count > 0:
    avg_confidence = confidence_sum / person_count
else:
    avg_confidence = 0

if person_count <= 2:
    risk_level = "Low"

elif person_count <= 5:
    risk_level = "Medium"

else:
    risk_level = "High"

print("\n----- Detection Summary -----")
print(f"People detected : {person_count}")
print(f"Average confidence : {avg_confidence:.2f}")
print(f"Risk level : {risk_level}")

if person_count > 0:
    print("ALERT: Potential survivors detected")

print("Output saved to outputs/result.jpg")

with open("detection_log.txt", "a") as log:
    log.write(
        f"{datetime.now()} | "
        f"People: {person_count} | "
        f"Confidence: {avg_confidence:.2f} | "
        f"Risk: {risk_level}\n"
    )