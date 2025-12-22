# drone_aim.py - OPTIMIZED FOR PERFORMANCE
import time
import cv2
import numpy as np
import serial
import serial.tools.list_ports
from ultralytics import YOLO
import cvzone
import Angles

# ------------------------- CONFIG --------------------------------
MODEL_PATH = "best.pt"
DRONE_CLASS_NAME = "drone"
CONF_THRESH = 0.35
DETECTION_SKIP = 3            # Process every 4th frame (0,1,2,3 skip, 4th process)
SCALE_Z = 1400.0
EMA_ALPHA = 0.15
SERIAL_BAUD = 115200

# Performance settings
FRAME_WIDTH = 640             # Reduced from 1280
FRAME_HEIGHT = 480            # Reduced from 720
TARGET_FPS = 30               # Increased for smoother display
INFERENCE_SIZE = 416          # YOLOv8 inference size (smaller = faster)
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
# Force CPU or GPU based on availability
# model.to('cuda')  # Uncomment if you have NVIDIA GPU
print("Model loaded")
# -----------------------------------------------------------------

# ------------------------- CAMERA SETUP ---------------------------
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffer lag

frame_idx = 0
ema_thetaX = None
ema_thetaY = None
last_detection = None  # Cache last detection

# Screen center adjusted for new resolution
scrn_cx = FRAME_WIDTH // 2
scrn_cy = FRAME_HEIGHT // 2

# Helper: get class name map
try:
    names = model.names
except Exception:
    names = {}

# Find drone class index
drone_class_idx = None
for k, v in names.items():
    if isinstance(v, str) and v.lower() == DRONE_CLASS_NAME.lower():
        drone_class_idx = k
        break

print("Model class names:", names)
print("Using drone class index:", drone_class_idx)
print(f"Resolution: {FRAME_WIDTH}x{FRAME_HEIGHT}, Processing every {DETECTION_SKIP+1} frames")

# FPS counter
fps_start_time = time.time()
fps_counter = 0
current_fps = 0

# ------------------------- MAIN LOOP ------------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        print("Camera read failed")
        break

    frame_idx += 1
    display_frame = frame.copy()

    # FPS calculation
    fps_counter += 1
    if fps_counter >= 30:
        fps_end_time = time.time()
        current_fps = fps_counter / (fps_end_time - fps_start_time)
        fps_start_time = fps_end_time
        fps_counter = 0

    # Run detection only on selected frames
    if frame_idx % (DETECTION_SKIP + 1) == 0:
        # Run inference with reduced image size for speed
        results = model(frame, imgsz=INFERENCE_SIZE, verbose=False)[0]
        
        boxes = []
        scores = []
        cls_ids = []

        if hasattr(results, "boxes") and results.boxes is not None:
            xyxy = results.boxes.xyxy.cpu().numpy()
            confs = results.boxes.conf.cpu().numpy()
            clss = results.boxes.cls.cpu().numpy().astype(int)

            for i, bb in enumerate(xyxy):
                x1, y1, x2, y2 = bb
                conf = float(confs[i])
                cls = int(clss[i])
                
                if conf < CONF_THRESH:
                    continue
                    
                if drone_class_idx is not None:
                    if cls != drone_class_idx:
                        continue
                
                boxes.append((int(x1), int(y1), int(x2), int(y2)))
                scores.append(conf)
                cls_ids.append(cls)

        # Choose the "closest" detection (largest height)
        chosen_box = None
        if boxes:
            best_idx = max(range(len(boxes)), key=lambda i: boxes[i][3] - boxes[i][1])
            chosen_box = boxes[best_idx]
            last_detection = chosen_box  # Cache for interpolation
        elif last_detection is not None:
            # Use cached detection if no new detection (smoother tracking)
            chosen_box = last_detection
    else:
        # Use cached detection on skipped frames
        chosen_box = last_detection

    # Process detection
    if chosen_box is not None:
        x1, y1, x2, y2 = chosen_box
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        bw = x2 - x1
        bh = y2 - y1

        # Compute coordinates
        X_virtual = -(cx - scrn_cx)
        Y_virtual = -(cy - scrn_cy)

        if bh > 0:
            Z_real = SCALE_Z / float(bh)
        else:
            Z_real = 0.0

        distance_virtual = bh if bh > 0 else 1.0
        X_real = X_virtual * (6.3 / distance_virtual)
        Y_real = Y_virtual * (6.3 / distance_virtual)

        # Draw on frame
        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.circle(display_frame, (int(cx), int(cy)), 4, (0, 0, 255), -1)
        cvzone.putTextRect(display_frame, f"Drone Z~{int(Z_real)}cm", (x1, y1 - 10), scale=0.8, thickness=1)

        # Angle calculation
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

        # Send commands
        cmdX = f"X{ema_thetaX}\n"
        cmdY = f"Y{ema_thetaY}\n"
        safe_write(serialcomm, cmdX)
        safe_write(serialcomm, cmdY)

    else:
        cvzone.putTextRect(display_frame, "No drone detected", (20, 30), scale=0.8, thickness=1)

    # Show FPS
    cvzone.putTextRect(display_frame, f"FPS: {int(current_fps)}", (20, FRAME_HEIGHT - 20), scale=0.8, thickness=1)

    # Display
    cv2.imshow("Drone Aimbot", display_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
serialcomm.close()
