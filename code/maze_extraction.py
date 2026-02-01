import cv2
import numpy as np
import json
import os

# ==============================
# CONFIG (EDIT IF NEEDED)
# ==============================

VIDEO_PATH = "input/longmaze_solved.mp4"
OUTPUT_PATH = "output/longmaze_trace.json"

GRID_ROWS = 23        # number of rows in maze
GRID_COLS = 23        # number of columns in maze
FRAME_STEP = 10        # sample every N frames

# red color range in HSV (agent)
RED_LOWER = np.array([0, 70, 50])
RED_UPPER = np.array([10, 255, 255])


# ==============================
# UTILS
# ==============================

def deduplicate_consecutive(seq):
    result = []
    for s in seq:
        if not result or result[-1] != s:
            result.append(s)
    return result

def cell_to_action(c1, c2):
    x1, y1 = c1
    x2, y2 = c2
    if x2 == x1 + 1 and y2 == y1:
        return "Right"
    if x2 == x1 - 1 and y2 == y1:
        return "Left"
    if y2 == y1 + 1 and x2 == x1:
        return "Down"
    if y2 == y1 - 1 and x2 == x1:
        return "Up"
    return "Unknown"

# ==============================
# MAIN PIPELINE
# ==============================

def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError("Cannot open video")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    cell_w = width / GRID_COLS
    cell_h = height / GRID_ROWS

    trajectory = []
    frame_id = 0
    total_turns=0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_id % FRAME_STEP != 0:
            frame_id += 1
            continue

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, RED_LOWER, RED_UPPER)

        moments = cv2.moments(mask)
        if moments["m00"] > 0:
            cx = int(moments["m10"] / moments["m00"])
            cy = int(moments["m01"] / moments["m00"])

            grid_x = int(cx / cell_w)
            grid_y = int(cy / cell_h)

            trajectory.append([grid_x, grid_y])
            total_turns=total_turns+1

        frame_id += 1

    cap.release()

    # Deduplicate repeated positions
    trajectory = deduplicate_consecutive(trajectory)

    # Extract actions
    actions = []
    for i in range(len(trajectory) - 1):
        actions.append(cell_to_action(trajectory[i], trajectory[i + 1]))

    # Save output
    os.makedirs("output", exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(
            {
                "trajectory": trajectory,
                "actions": actions
            },
            f,
            indent=2
        )

    print("✅ Extraction complete")
    print("Trajectory:", trajectory)
    print("Actions:", actions)
    print("total_turns: ", total_turns)

# ==============================
# ENTRY POINT
# ==============================

if __name__ == "__main__":
    main()
