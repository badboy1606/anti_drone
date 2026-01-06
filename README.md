# Anti-Drone System (Proof of Concept)

## Background

The **Indo-Tibetan Border Police (ITBP)** is deployed along the Indo-China border, characterized by **high-altitude mountainous terrain**, **remote and sparsely populated regions**, and **extreme environmental conditions**.  
Sub-zero temperatures, low atmospheric pressure, and high wind speeds significantly degrade the performance of both **human operators** and **electronic equipment** compared to plains and semi-plains.

These regions are highly vulnerable to **enemy drone intrusions**, necessitating an effective **counter-drone solution** capable of **early detection** and **soft-kill neutralization** (jamming and spoofing).  
The objective is to establish a **protective electronic umbrella** within a safe operational radius to detect, track, and neutralize hostile drones.

---

## Problem Statement (Competition Brief)

An anti-drone system is required with **portable detection, jamming, and spoofing modules**, capable of countering **single and swarm drone threats** approaching simultaneously from multiple directions.

### Required Capabilities

### A) System Components
- RF Detector (Wideband Scanner) for drone detection  
- RF and Satellite Navigation Jamming System  
- Optional Drone Spoofing / Takeover Capability  

### B) Detection Range
- Micro & Small Drones: **5 km or better**  
- Medium Drones: **10 km or better**

### C) Jamming Range
- Omni-directional: **≥ 3 km**  
- Directional: **≥ 4 km**

### D) Spoofing Range (ISM Bands: 400 MHz – 8 GHz)
- Omni-directional: **≥ 20 km**  
- Directional: **≥ 40 km**

---

## Competition Context

This project was developed as part of the **Smart India Hackathon (SIH)**, focusing on **defence and national security applications**.  
The aim was to design and demonstrate a **feasible, modular, and scalable counter-drone architecture**, emphasizing **concept validation and system integration** rather than full-scale deployment.

---

## Project Scope

⚠️ **Important Note:**  
This system is a **Proof of Concept (PoC)** implementation of the above problem statement.

- Demonstrates **core principles** of detection, jamming, and spoofing  
- Validates feasibility at **prototype and simulation level**  
- Does **not claim to achieve operational field ranges** mentioned in the problem statement  
- Designed for **academic, research, and controlled testing environments**

---

## System Overview

The Anti-Drone System is designed as a **modular architecture**, allowing independent development and evaluation of each subsystem.

---

## System Modules

### 📡 Drone Detection (HackRF)

Drone detection is performed using **Software Defined Radio (SDR)** techniques with **HackRF**, focusing on identifying RF activity associated with drone communication and control links.

> This module serves as a detection PoC.  
> Detailed implementation and enhancements are maintained by the contributors working specifically on detection.

📂 **Module directory:** `detection/`

---

### 📡 Jamming

The jamming module explores **soft-kill techniques** to disrupt drone control and navigation links by transmitting interference signals on targeted frequency bands.

➡️ **Read detailed documentation:**  
🔗 [Jamming Module README](jamming/)

---

### 🛰️ Spoofing

The spoofing module demonstrates a **prototype GNSS spoofing mechanism** using ESP32 microcontrollers, highlighting vulnerabilities in unauthenticated navigation systems by transmitting false but valid-looking coordinate data.

➡️ **Read detailed documentation:**  
🔗 [Spoofing Module README](spoofing/)

---

## Our Approach

1. **Threat Understanding**  
   Studied drone communication, navigation methods, and vulnerabilities relevant to border surveillance scenarios.

2. **Modular Design**  
   Split the system into detection, jamming, and spoofing modules for independent development and testing.

3. **Soft-Kill Focus**  
   Prioritized non-kinetic neutralization methods (jamming & spoofing) suitable for sensitive and remote environments.

4. **Prototype Validation**  
   Implemented PoC-level demonstrations using SDRs and microcontrollers to validate concepts.

5. **Scalability Consideration**  
   Designed architecture such that higher-power RF front-ends and directional systems can be integrated in future iterations.

---

## Project Structure

```text
anti-drone-system/
├── README.md
├── detection/
├── jamming/
│   └── README.md
├── spoofing/
│   └── README.md
