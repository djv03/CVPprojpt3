import cv2
import numpy as np
import json

VIDEO_PATH = "smartphone_stacking.mp4"

# ---------- Utility ----------
def get_centroid_and_area(mask):
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None, None
    c = max(cnts, key=cv2.contourArea)
    area = cv2.contourArea(c)
    M = cv2.moments(c)
    if M["m00"] == 0:
        return None, None
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    return (cx, cy), area

# ---------- Load Video ----------
cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)

states = []
frame_idx = 0

# ---------- HSV ranges (tuned for your video) ----------
# Brown phone (right, moving)
brown_lower = np.array([5, 50, 50])
brown_upper = np.array([20, 255, 255])

# Greenish phone (left, stationary)
green_lower = np.array([35, 40, 40])
green_upper = np.array([85, 255, 255])

# ---------- Process ----------
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Sample every 3 frames (fast & stable)
    if frame_idx % 3 != 0:
        frame_idx += 1
        continue

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    mask_brown = cv2.inRange(hsv, brown_lower, brown_upper)
    mask_green = cv2.inRange(hsv, green_lower, green_upper)

    (cR, areaR) = get_centroid_and_area(mask_brown)
    (cL, areaL) = get_centroid_and_area(mask_green)
    print("cntroid R: ", cR)
    print("cntroid L: ", cL)
    

    if cR and cL:
        states.append({
            "t": round(frame_idx / fps, 2),
            "PhoneR": {"x": cR[0], "y": cR[1], "area": int(areaR)},
            "PhoneL": {"x": cL[0], "y": cL[1], "area": int(areaL)}
        })

    frame_idx += 1
    print("current frame ",frame_idx)

cap.release()

# ---------- Event Extraction ----------
events = []
areas = [s["PhoneR"]["area"] for s in states]
positions = [(s["PhoneR"]["x"], s["PhoneR"]["y"]) for s in states]

def moved(p1, p2, th=15):
    return abs(p1[0]-p2[0]) + abs(p1[1]-p2[1]) > th

if states:
    events.append("Unstacked")

    # Lift detection (area increase)
    if max(areas) > areas[0] * 1.1:
        events.append("Lift")

    # Move detection
    for i in range(1, len(positions)):
        if moved(positions[i-1], positions[i]):
            events.append("Move")
            break

    # Place detection (motion stops)
    if not moved(positions[-2], positions[-1]):
        events.append("Place")

    # Stacked detection (occlusion / overlap)
    last = states[-1]
    dx = abs(last["PhoneR"]["x"] - last["PhoneL"]["x"])
    dy = abs(last["PhoneR"]["y"] - last["PhoneL"]["y"])
    if dx < 200 and dy < 200:
        events.append("Stacked")

# Remove duplicates
events = list(dict.fromkeys(events))

# ---------- Output ----------
output = {
    "states": states,
    "events": events,
    "final_relation": "On(PhoneR, PhoneL)" if "Stacked" in events else "Unstacked"
}

with open("output.json", "w") as f:
    json.dump(output, f, indent=2)

print("Extraction complete.")
print("Events:", events)
