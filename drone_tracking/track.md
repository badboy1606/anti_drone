# Drone Tracking & Aiming Module

This directory contains the **vision-based drone tracking and auto-aiming module**.


## Directory Structure
```
drone_tracking/
├── firmware/
│ └── main.c # ESP32 firmware for turret control
├── scripts/
│ ├── drone_aim.py # Main detection, tracking, and serial 
│ ├── aim.py # initial test for hand tracking (if used)
│ └── Angles.py # Pan–tilt angle computation
└── track.md # Module documentation
```


## Module Overview
- Detects drones from live camera feed using YOLO
- Estimates relative X, Y, Z position from bounding box geometry
- Computes turret pan and tilt angles
- Sends smoothed angle commands to ESP32 via serial

## Data Flow
Camera > Detection > Position Estimation > Angle Computation > ESP32

## Notes
- Optimized for real-time CPU performance
- Uses frame skipping and EMA smoothing
- Distance estimation is relative and vision-based
- ESP32 firmware is hardware-dependent and configurable

## Running the module 
- Flash main.c in the esp32
- Connect the servo motors correctly 
- Run drone_aim.py and see the turret correctly following the drone 
