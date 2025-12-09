# drone_aim.py
import time
import cv2
import numpy as np
import serial
import serial.tools.list_ports
from ultralytics import YOLO    # YOLOv8
import cvzone
import Angles

# ------------------------- CONFIG --------------------------------
MODEL_PATH = "best.pt"        # your YOLOv8 model
DRONE_CLASS_NAME = "drone"    # class name used in your dataset
CONF_THRESH = 0.35            # minimum detection confidence
DETECTION_SKIP = 1            # process every frame, increase to skip frames for speed
SCALE_Z = 1400.0              # calibration constant for Z estimation (tweak for your camera)
EMA_ALPHA = 0.15              # smoothing factor for angle smoothing (0..1)
SERIAL_BAUD = 115200
# -----------------------------------------------------------------

# ------------------------- SAFE SERIAL ----------------------------
def safe_write(port, data):
    try:
        port.write(data.encode())
    except serial.SerialException:
        print("⚠ Serial port crashed — reopening...")
        try:
            port.close()
        except:
            pass
        time.sleep(0.5)
        try:
            port.open()
            time.sleep(0.5)
            port.write(data.encode())
        except Exception as e:
            print("✖ Failed to reopen serial:", e)
# -----------------------------------------------------------------

# ------------------------- AUTO-DETECT PORT -----------------------
def find_esp32_port():
    ports = serial.tools.list_ports.comports()
    for p in ports:
        name = p.description.lower()
        if any(chip in name for chip in ["cp210", "silicon", "ch340", "usb"]):
            return p.device
    return None
# -----------------------------------------------------------------

# ------------------------- SETUP SERIAL ---------------------------
port = find_esp32_port()
if port is None:
    print("❌ ESP32 not detected!")
    raise SystemExit

serialcomm = serial.Serial(port=port, baudrate=SERIAL_BAUD, timeout=1)
time.sleep(0.5)
serialcomm.flush()
print(f"🔌 ESP32 detected on: {port}")
# -----------------------------------------------------------------

# ------------------------- LOAD MODEL -----------------------------
model = YOLO(MODEL_PATH)
# If you want GPU set device when creating model or via model.predict(device='cuda:0') etc.
# -----------------------------------------------------------------

# ------------------------- CAMERA SETUP ---------------------------
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 15)

frame_idx = 0
ema_thetaX = None
ema_thetaY = None

# Helper: get class name map
try:
    names = model.names  # dict: idx -> name
except Exception:
    names = {}

# Reverse lookup: class index for DRONE_CLASS_NAME (if available)
drone_class_idx = None
for k, v in names.items():
    if isinstance(v, str) and v.lower() == DRONE_CLASS_NAME.lower():
        drone_class_idx = k
        break

# If class index not found, we'll filter by name from results if present
print("Model class names:", names)
print("Using drone class index:", drone_class_idx)

# ------------------------- MAIN LOOP ------------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        print("Camera read failed")
        break

    frame_idx += 1
    display_frame = frame.copy()

    # Run detection (optionally skip frames for performance)
    if frame_idx % (DETECTION_SKIP + 1) == 0:
        # YOLOv8 returns a Results object list — call model(frame) and use the first result
        results = model(frame)[0]            # single-frame inference
        boxes = []
        scores = []
        cls_ids = []

        if hasattr(results, "boxes") and results.boxes is not None:
            # .boxes.xyxy, .boxes.conf, .boxes.cls
            xyxy = results.boxes.xyxy.cpu().numpy()    # shape (N,4)
            confs = results.boxes.conf.cpu().numpy()   # shape (N,)
            clss = results.boxes.cls.cpu().numpy().astype(int)  # shape (N,)

            for i, bb in enumerate(xyxy):
                x1, y1, x2, y2 = bb
                conf = float(confs[i])
                cls = int(clss[i])
                # Filter by confidence
                if conf < CONF_THRESH:
                    continue
                # If drone_class_idx known, filter by it. Otherwise accept any and later check name.
                if drone_class_idx is not None:
                    if cls != drone_class_idx:
                        continue
                # Save
                boxes.append((int(x1), int(y1), int(x2), int(y2)))
                scores.append(conf)
                cls_ids.append(cls)

        # If class idx unknown, fall back to textual classes (ultralytics may place class names in results.names)
        # Already filtered when drone_class_idx is set; otherwise we could check results.names mapping.

        # Choose the "closest" detection — pick the bbox with largest height (y2-y1)
        chosen_box = None
        if boxes:
            best_idx = None
            best_h = -1
            for i, (x1, y1, x2, y2) in enumerate(boxes):
                h = (y2 - y1)
                if h > best_h:
                    best_h = h
                    best_idx = i
            chosen_box = boxes[best_idx]

        # If we found one, compute coordinates + send commands
        if chosen_box is not None:
            x1, y1, x2, y2 = chosen_box
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            bw = x2 - x1
            bh = y2 - y1

            # Compute "real" X/Y relative to screen center (same logic as your hand code)
            scrn_cx = 512
            scrn_cy = 288
            X_virtual = -(cx - scrn_cx)
            Y_virtual = -(cy - scrn_cy)

            # Estimate Z from bbox height (inverse relation) — calibrate SCALE_Z for your camera/setup.
            # Larger bbox height => closer drone => smaller Z value.
            if bh > 0:
                Z_real = SCALE_Z / float(bh)   # crude approximation in cm; calibrate SCALE_Z
            else:
                Z_real = 0.0

            # Convert virtual->real using same scaling you used earlier for hand (6.3/distance_virtual)
            # For drones distance_virtual replaced by bbox height proxy to avoid divide by zero.
            # Use bh as the proxy for "distance virtual"
            distance_virtual = bh if bh > 0 else 1.0
            X_real = X_virtual * (6.3 / distance_virtual)
            Y_real = Y_virtual * (6.3 / distance_virtual)

            # Display box + center on frame
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(display_frame, (int(cx), int(cy)), 4, (0, 0, 255), -1)
            cvzone.putTextRect(display_frame,
                               f"drone {int(bh)}px Z~{int(Z_real)}cm",
                               (x1, y1 - 10))

            # ANGLE CALC using your Angles.turret
            angles = Angles.turret(X_real, Y_real, Z_real)
            angles.offsets(12, 0, 7)
            angles.getAngles()

            raw_thetaX = int(angles.getTheta_x()) + 10
            raw_thetaY = int(angles.getTheta_y()) + 3

            # EMA smoothing
            if ema_thetaX is None:
                ema_thetaX = raw_thetaX
                ema_thetaY = raw_thetaY
            else:
                ema_thetaX = int((1 - EMA_ALPHA) * ema_thetaX + EMA_ALPHA * raw_thetaX)
                ema_thetaY = int((1 - EMA_ALPHA) * ema_thetaY + EMA_ALPHA * raw_thetaY)

            # Prepare commands
            cmdX = f"X{ema_thetaX}\n"
            cmdY = f"Y{ema_thetaY}\n"

            # Send via serial
            safe_write(serialcomm, cmdX)
            safe_write(serialcomm, cmdY)

            # debug prints (optional)
            # print(f"Sent -> {cmdX.strip()}, {cmdY.strip()}  (bh={bh})")

        else:
            # No drone found: optional behaviour
            cvzone.putTextRect(display_frame, "No drone", (20, 30))
            # Optionally you can send a 'stop' or center command or skip sending
            # Example: send a small heartbeat every few seconds (not required)
    # else: skipping this frame for speed

    # Show frame
    cv2.imshow("Drone Aimbot", display_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# cleanup
cap.release()
cv2.destroyAllWindows()
serialcomm.close()
