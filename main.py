import cv2
import time
import csv
import os
import pandas as pd
import streamlit as st
from ultralytics import YOLO
from streamlit_autorefresh import st_autorefresh

# -----------------------------
# Settings
# -----------------------------
snapshot_interval = 2  # seconds between snapshots
csv_filename = "queue_log.csv"

# -----------------------------
# Initialize YOLO model
# -----------------------------
model = YOLO("yolov8n.pt")

# -----------------------------
# Setup Streamlit
# -----------------------------
st.set_page_config(page_title="Restaurant Queue Monitor", layout="wide")
st.title("Live Queue Length")

# Camera
cap = cv2.VideoCapture(0)

# Create CSV if not exists
if not os.path.exists(csv_filename):
    with open(csv_filename, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["People_in_Line"])

# Auto-refresh every N seconds
st_autorefresh(interval=snapshot_interval * 1000, key="queue_refresh")

# Take one frame and update count
ret, frame = cap.read()
if ret:
    results = model(frame)
    people_count = sum([1 for cls in results[0].boxes.cls if int(cls) == 0])

    # Save to CSV
    with open(csv_filename, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([people_count])

    # Display big live number only
    st.metric("People in Line", people_count)

# Release camera when done
cap.release()