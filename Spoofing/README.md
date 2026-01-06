# Spoofing

## Overview

**Spoofing** is a signal-level attack in which false but valid-looking data is transmitted to deceive a target system into accepting incorrect information.  
In anti-drone systems, GNSS spoofing is used to manipulate a drone’s perceived position, velocity, or altitude by transmitting counterfeit navigation data.

Unlike jamming, spoofing does not deny service outright; instead, it **misleads the system while maintaining signal validity**.

---

## GNSS Spoofing Using ESP32

This module demonstrates a **prototype-scale GNSS spoofing mechanism** implemented using ESP32 microcontrollers. The system emulates GNSS-like coordinate data and transmits it wirelessly to a drone-side receiver, causing the drone to compute incorrect position information.

The goal of this implementation is **conceptual demonstration and educational validation**, not real-world deployment.

---

## System Architecture

- **Spoofing Node (ESP32 – Transmitter)**
  - Generates predefined or dynamic fake GPS coordinates
  - Broadcasts data wirelessly using ESP-NOW / Wi-Fi
  - Simulates satellite-originated navigation messages

- **Drone Node (ESP32 – Receiver)**
  - Receives incoming coordinate packets
  - Accepts the first valid signal as the navigation source
  - Updates perceived position based on received data

---

## Working Principle

1. The spoofing ESP32 transmits fabricated latitude, longitude, and altitude data.
2. The receiver ESP32 treats the spoofed signal as a valid navigation source.
3. If spoofing is initiated before a legitimate signal is locked, the system prioritizes the spoofed coordinates.
4. The drone computes its position using the spoofed data, resulting in incorrect localization.

---

## Key Observations

- The receiver locks onto the **first strong and consistent signal**
- Absence of authentication makes GNSS systems vulnerable
- Sudden coordinate jumps indicate spoofing activity
- Multi-source verification can reduce susceptibility

---

## Limitations

- This implementation does **not generate real RF GNSS signals**
- No cryptographic authentication is bypassed
- Intended only for **controlled lab environments**
- Accuracy depends on packet timing and transmission consistency

---

## Defensive Relevance

Understanding spoofing mechanisms enables the development of effective countermeasures such as:

- Signal consistency checks
- Multi-constellation verification
- Inertial sensor fusion
- Sudden trajectory anomaly detection

---

## Disclaimer

This project is developed **strictly for academic, educational, and defensive research purposes**.  
No part of this work is intended for misuse or unauthorized interference with real-world navigation systems.
