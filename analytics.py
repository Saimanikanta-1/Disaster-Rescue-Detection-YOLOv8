import pandas as pd

data = {
    "Metric": [
        "Frames Processed",
        "Total Detections"
    ],
    "Value": [
        313,
        4222
    ]
}

df = pd.DataFrame(data)

df.to_csv("outputs/detection_report.csv", index=False)

print("Report saved to outputs/detection_report.csv")