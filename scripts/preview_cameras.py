#!/usr/bin/env python3
"""Quick camera preview — shows all connected cameras so you can identify them.

Usage:
    python3 scripts/preview_cameras.py
"""

import cv2
import sys

DEVICE_INDICES = [0, 2, 4]

caps = {}
for idx in DEVICE_INDICES:
    cap = cv2.VideoCapture(idx)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            caps[idx] = cap
            print(f"video{idx}: opened ({frame.shape[1]}x{frame.shape[0]})")
        else:
            cap.release()
            print(f"video{idx}: opened but no frame")
    else:
        print(f"video{idx}: not available")

if not caps:
    print("No cameras found.")
    sys.exit(1)

print()
print("Camera windows are open. Look at each window and note which")
print("physical camera it is (overhead, arm_head, or bottom).")
print("Press 'q' to quit.")

while True:
    for idx, cap in caps.items():
        ret, frame = cap.read()
        if not ret:
            continue
        # Shrink large frames so they fit on screen
        h, w = frame.shape[:2]
        if w > 640:
            scale = 640 / w
            frame = cv2.resize(frame, (640, int(h * scale)))
        cv2.putText(frame, f"/dev/video{idx}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.imshow(f"video{idx}", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

for cap in caps.values():
    cap.release()
cv2.destroyAllWindows()
