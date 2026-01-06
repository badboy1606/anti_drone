# Drone Detection

## Overview

Drone Detection focuses on identifying the presence of unmanned aerial vehicles (UAVs) by analyzing their radio-frequency (RF) signatures rather than relying on visual or acoustic cues.
Consumer and commercial drones emit distinctive RF patterns due to Wi-Fi control links, telemetry, and frequency-hopping behavior.

This module implements a signal-level RF drone detection pipeline using Software Defined Radio (SDR), mathematical signal analysis, and vision-language models (VLMs) for spectral interpretation.

---

## Detection Stack
The detection system is composed of four tightly integrated components:
• IQ Signal Analysis (Offline)
• Live RF Monitoring (Online)
• Waterfall Graph Intelligence
• Web-Based Control & Visualization Interface
Each component validates drone presence from a different perspective, increasing robustness and reducing false positives.

---

## IQ Signal Analysis (iq_analyser.py)
• Processes recorded IQ (In-phase & Quadrature) data files
• Performs spectral and temporal analysis
• Detects RF patterns characteristic of drone communication links
• Outputs a binary decision: Drone Present / Not Present

### Key Capabilities
• Power spectral density (PSD) estimation
• Band occupancy detection
• Noise vs structured-signal discrimination
This module is used for post-mission analysis, dataset validation, and model testing.

---

## Live RF Monitoring (signal_watch.py)
• Collects real-time RF data from an SDR
• Continuously monitors frequency bands commonly used by drones
• Differentiates drone Wi-Fi signals from regular Wi-Fi traffic

### Core Logic
• Mathematical analysis of frequency hopping patterns
• Temporal consistency checks
• Bandwidth and dwell-time estimation
• Statistical deviation from standard Wi-Fi behavior

### Output
• Live drone detection status
• Classification of signal type (Drone vs Non-drone Wi-Fi)

---

## Waterfall Graph Analyzer (VLM-based)
• Generates waterfall (time–frequency) spectrograms
• Analyzes them using a Vision-Language Model (Moondream VLM)
• Interprets visual RF patterns instead of raw signal values

### What It Detects
• Presence of structured RF activity
• Directional trends:
  -->Drone approaching
  -->Drone moving away
• Intensity changes over time
This approach enables human-like interpretation of RF spectrums, bridging signal processing and visual intelligence.\

---

## Web Interface
A unified web-based dashboard simplifies interaction with the entire detection stack.

### Features
• Start/stop live SDR monitoring
• View real-time detection results
• Display waterfall graphs with VLM-based inference
• Centralized status panel for all detection modules

### Benefits
• No CLI complexity
• Faster experimentation and debugging
• Operator-friendly deployment

---

## Key Observations
• Drone RF signals exhibit non-random, structured hopping
• Multi-layer verification significantly lowers false alarms
• Visual spectrogram analysis is surprisingly effective
• Multi-layer verification significantly lowers false alarms
• RF-based detection works even in low-visibility conditions

## Limitations
• Detection accuracy depends on SDR bandwidth and placement
• Encrypted payloads are not decoded—only RF behavior is analyzed
• High RF noise environments may require threshold tuning
• Waterfall interpretation is probabilistic, not absolute

## Defensive Relevance
• This module demonstrates how passive RF sensing can be used for:
• Early drone presence detection
• Airspace monitoring in restricted zones
• Complementing radar and vision-based systems
• It is especially relevant where low-cost, non-emissive detection is required.

## Disclaimer
This project is developed strictly for academic, educational, and defensive research purposes.
It does not decode private communications or interfere with RF transmissions and is intended only for controlled and authorized environments.