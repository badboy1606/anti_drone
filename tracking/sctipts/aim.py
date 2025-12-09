import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import cvzone
import Angles
import serial
import serial.tools.list_ports
import time

# ---------------------------------------------------------
# SAFE UART WRITE (fixes access denied error)
# ---------------------------------------------------------
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
        port.open()
        time.sleep(0.5)
        port.write(data.encode())

# ---------------------------------------------------------
# AUTO-DETECT ESP32 PORT
# ---------------------------------------------------------
def find_esp32_port():
    ports = serial.tools.list_ports.comports()
    for p in ports:
        name = p.description.lower()
        if any(chip in name for chip in ["cp210", "silicon", "ch340", "usb"]):
            return p.device
    return None


port = find_esp32_port()
if port is None:
    print("❌ ESP32 not detected!")
    exit()

print(f"🔌 ESP32 detected on: {port}")

serialcomm = serial.Serial(port=port, baudrate=115200, timeout=1)
time.sleep(1)
serialcomm.flush()

# ---------------------------------------------------------
# CAMERA SETUP
# ---------------------------------------------------------
camera = cv2.VideoCapture(0)
camera.set(3, 1280)
camera.set(4, 720)
camera.set(cv2.CAP_PROP_FPS, 15)

# ---------------------------------------------------------
# HAND TRACKER
# ---------------------------------------------------------
detector = HandDetector(
    detectionCon=0.6,
    maxHands=1,
    modelComplexity=0
)

# ---------------------------------------------------------
# DEPTH MODEL CONSTANTS
# ---------------------------------------------------------
A = 0.012636680507237852
B = -2.710541724316941
C = 182.62076069382988

# ---------------------------------------------------------
# MAIN LOOP
# ---------------------------------------------------------
while True:
    success, img = camera.read()
    hands, img = detector.findHands(img, draw=False)

    if hands:
        hand = hands[0]
        lmList = hand["lmList"]
        bx, by, bw, bh = hand["bbox"]

        # 3-value unpacking
        x1, y1, _ = lmList[5]
        x2, y2, _ = lmList[17]

        center_x = (x2 + x1) / 2
        center_y = (y2 + y1) / 2
        distance_virtual = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

        # Z Distance Model
        if distance_virtual > 153:
            Z_real = (-0.125 * distance_virtual) + 44.125
        elif 107 < distance_virtual < 153:
            Z_real = (-0.2173913 * distance_virtual) + 58.260869
        else:
            Z_real = A * distance_virtual**2 + B * distance_virtual + C

        # X-Y offsets
        screen_x = 512
        screen_y = 288

        X_virtual = -(center_x - screen_x)
        Y_virtual = -(center_y - screen_y)

        X_real = X_virtual * (6.3 / distance_virtual)
        Y_real = Y_virtual * (6.3 / distance_virtual)

        # Display
        cvzone.putTextRect(img, f"{int(X_real)} {int(Y_real)} {int(Z_real)}", (bx, by))

        # ANGLE CALC
        angles = Angles.turret(X_real, Y_real, Z_real)
        angles.offsets(12, 0, 7)
        angles.getAngles()

        thetaX = int(angles.getTheta_x()) + 10
        thetaY = int(angles.getTheta_y()) + 3

        # SEND TO ESP32 OVER USB UART0
        cmdX = f"X{thetaX}\n"
        cmdY = f"Y{thetaY}\n"

        safe_write(serialcomm, cmdX)
        safe_write(serialcomm, cmdY)

    cv2.imshow("Aimbot", img)
    cv2.waitKey(1)

# KEEP SERIAL PORT ALIVE (IMPORTANT!)
while True:
    time.sleep(1)

