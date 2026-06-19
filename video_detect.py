from ultralytics import YOLO
import cv2
import pandas as pd

model = YOLO("yolov8n.pt")

video_path = "videos/input.mp4"

cap = cv2.VideoCapture(video_path)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

output_video = cv2.VideoWriter(
"outputs/detected_video.mp4",
cv2.VideoWriter_fourcc(*"mp4v"),
fps,
(width, height)
)

total_people = 0
frame_count = 0
frame_data = []

while True:

    ret, frame = cap.read()

    if not ret:
        break

    results = model(frame)

    people_in_frame = 0

    for result in results:

        for box in result.boxes:

            class_id = int(box.cls[0])

            if class_id == 0:

                people_in_frame += 1

                confidence = float(box.conf[0])

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                label = f"Person {confidence:.2f}"

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    frame,
                    label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2
                )

    frame_data.append({
        "Frame": frame_count + 1,
        "People_Detected": people_in_frame
    })

    total_people += people_in_frame
    frame_count += 1

    output_video.write(frame)

cap.release()
output_video.release()

df = pd.DataFrame(frame_data)
df.to_csv("outputs/frame_analytics.csv", index=False)

print("Video processing completed")
print(f"Frames processed: {frame_count}")
print(f"Total detections: {total_people}")
print("Output saved to outputs/detected_video.mp4")
print("Analytics saved to outputs/frame_analytics.csv")
