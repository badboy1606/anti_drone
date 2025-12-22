###############################################
# MASTER RF + DRONE DETECTION INTEGRATION
# File: rf_master_drone_detector.py
###############################################

import time
import os
from iq_analyze import DroneDetector    # <-- Your FIRST script
from signal_watch import RFMonitor     # <-- Your SECOND script


def run_drone_detection_on_iq(iq_file, center_freq, sample_rate):
    """Runs DroneDetector on the newly captured IQ file."""
    
    print("\n🚀 Running Drone Detector on:", iq_file)

    detector = DroneDetector(
        filepath=iq_file,
        sample_rate=sample_rate,
        center_freq=center_freq,
        data_type="complex64",   # Change if your IQ capture uses int8
        max_samples=5_000_000,
        debug=True
    )

    drone_detected, results = detector.detect_drones()

    print("\n==============================")
    print(" DRONE DETECTION RESULT")
    print("==============================")
    if drone_detected:
        print("🚨 DRONE FOUND!")
    else:
        print("✓ No drone detected")
    print("==============================\n")

    return drone_detected, results


def main():
    print("\n===============================================")
    print("     MASTER RF + DRONE DETECTION SYSTEM")
    print("===============================================\n")

    monitor = RFMonitor()   # from your second script

    while True:
        print("\n📡 Waiting for an RF signal...")
        detection = monitor.run_single_detection()

        if not detection:
            print("⚠ No signal captured – retrying...")
            time.sleep(1)
            continue

        # -----------------------------
        # 1️⃣ IQ FILE IS NOW READY
        # -----------------------------
        iq_file = detection.iq_filename
        freq = detection.frequency
        print("\n📁 New IQ file created:", iq_file)

        # -----------------------------
        # 2️⃣ CALL DRONE ANALYZER
        # -----------------------------
        run_drone_detection_on_iq(
            iq_file=iq_file,
            center_freq=freq,
            sample_rate=monitor.config.sample_rate
        )

        # -----------------------------
        # 3️⃣ WAIT FOR USER DECISION
        # -----------------------------
        monitor.wait_for_continue()


if __name__ == "__main__":
    main()
