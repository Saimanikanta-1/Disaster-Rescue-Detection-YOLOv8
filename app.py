import streamlit as st
from ultralytics import YOLO
import cv2
import tempfile

st.set_page_config(page_title="Disaster Survivor Detection")

st.title("AI-Based Disaster Survivor Detection System")

st.write(
"Upload an image to detect potential survivors using YOLOv8."
)

model = YOLO("yolov8n.pt")

uploaded_file = st.file_uploader(
"Upload an Image",
type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    temp_file = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".jpg"
    )

    temp_file.write(uploaded_file.read())
    temp_file.close()

    image = cv2.imread(temp_file.name)

    if image is None:
        st.error("Failed to load image.")
    else:
        results = model(image)

        person_count = 0

        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0])

                if class_id == 0:
                    person_count += 1
                    confidence = float(box.conf[0])
                    x1, y1, x2, y2 = map(
                        int,
                        box.xyxy[0]
                    )

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

        if person_count <= 2:
            risk_level = "Low"
        elif person_count <= 5:
            risk_level = "Medium"
        else:
            risk_level = "High"

        st.subheader("Detection Results")
        st.metric("People Detected", person_count)
        st.metric("Risk Level", risk_level)

        rgb_image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        st.image(
            rgb_image,
            caption="Detection Output",
            use_container_width=True
        )

        output_path = "outputs/streamlit_result.jpg"
        cv2.imwrite(output_path, image)

        with open(output_path, "rb") as file:
            st.download_button(
                label="Download Result",
                data=file,
                file_name="detected_result.jpg",
                mime="image/jpeg"
            )

