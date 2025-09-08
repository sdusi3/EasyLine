import cv2
import time
import csv
from ultralytics import YOLO

# -----------------------------
# 1. Load YOLOv8 Model
# -----------------------------
model = YOLO("yolov8n.pt")  # 'n' = nano version, fast for snapshots

# -----------------------------
# 2. Connect to Camera
# -----------------------------
cap = cv2.VideoCapture(0)  # 0 = default webcam; replace with RTSP/URL if IP camera

# -----------------------------
# 3. Set Snapshot Interval
# -----------------------------
snapshot_interval = 5  # seconds between snapshots

# -----------------------------
# 4. CSV Setup
# -----------------------------
csv_filename = "queue_log.csv"

# Create CSV and write header if file doesn't exist
with open(csv_filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Timestamp", "People_in_Line"])

# -----------------------------
# 5. Run Loop
# -----------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Run YOLO detection
    results = model(frame)

    # Count people (class 0 = 'person')
    people_count = sum([1 for cls in results[0].boxes.cls if int(cls) == 0])

    # Timestamp
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

    # Print and log to CSV
    print(f"{timestamp} | People in line: {people_count}")
    with open(csv_filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([timestamp, people_count])

    # Optional: display annotated frame
    annotated_frame = results[0].plot()
    cv2.imshow("YOLO Snapshot Detection", annotated_frame)

    # Wait until next snapshot
    if cv2.waitKey(snapshot_interval * 1000) & 0xFF == ord("q"):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()