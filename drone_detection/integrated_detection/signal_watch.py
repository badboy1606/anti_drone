import subprocess
import time
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from typing import Optional, Tuple


class RFConfig:
    """Configuration settings for RF monitoring"""
    def __init__(self):
        self.threshold = -55
        self.sweep_range = "1000:6000"
        self.bin_width = "400000"
        self.sample_rate = 20e6
        self.capture_seconds = 2
        self.fft_size = 2048
        self.dynamic_range_db = 50
        
    @property
    def num_samples(self) -> int:
        return int(self.sample_rate * self.capture_seconds)


class ColorMapGenerator:
    """Generates custom colormaps for waterfall display"""
    
    @staticmethod
    def create_sdr_colormap() -> LinearSegmentedColormap:
        """Creates Cyan/Yellow/Orange/Red colormap matching classic SDR style"""
        colors = [
            (0.0, 0.4, 0.5),   # Dark Cyan (noise floor)
            (0.0, 0.8, 0.8),   # Bright Cyan (weak signal)
            (0.4, 1.0, 0.6),   # Cyan-Green transition
            (1.0, 1.0, 0.0),   # Yellow (medium signal)
            (1.0, 0.6, 0.0),   # Orange (strong signal)
            (1.0, 0.3, 0.0),   # Deep Orange (very strong)
            (1.0, 0.0, 0.0)    # Red (peak)
        ]
        return LinearSegmentedColormap.from_list("SDR_CyanYellowRed", colors, N=256)


class HackRFSweep:
    """Manages HackRF sweep operations"""
    
    def __init__(self, config: RFConfig):
        self.config = config
        self.process: Optional[subprocess.Popen] = None
        
    def start(self) -> subprocess.Popen:
        """Start HackRF sweep process"""
        self.process = subprocess.Popen(
            ["hackrf_sweep", "-f", self.config.sweep_range, "-w", self.config.bin_width],
            stdout=subprocess.PIPE
        )
        return self.process
    
    def stop(self):
        """Stop sweep process"""
        if self.process:
            self.process.terminate()
            time.sleep(0.5)
            
    def parse_sweep_line(self, line: str) -> Optional[Tuple[float, float, list]]:
        """Parse a sweep line and return (freq_start, bin_width, powers)"""
        if not line or "sweeps" in line:
            return None
            
        parts = line.replace(",", "").split()
        
        try:
            freq_start = float(parts[2])
            bin_width = float(parts[4])
            powers = [float(p) for p in parts[6:]]
            return (freq_start, bin_width, powers)
        except (IndexError, ValueError):
            return None


class IQCapture:
    """Handles IQ data capture from HackRF"""
    
    def __init__(self, config: RFConfig):
        self.config = config
        
    def capture(self, frequency: int, filename: str) -> bool:
        """Capture IQ data at specified frequency"""
        record_cmd = [
            "hackrf_transfer",
            "-f", str(frequency),
            "-s", str(int(self.config.sample_rate)),
            "-n", str(self.config.num_samples),
            "-r", filename
        ]
        
        subprocess.run(record_cmd)
        return os.path.exists(filename)
    
    def load_iq_file(self, filename: str) -> Optional[np.ndarray]:
        """Load and parse IQ file"""
        try:
            raw = np.fromfile(filename, dtype=np.int8)
            raw = raw[:len(raw) - (len(raw) % 2)]
            iq = raw[0::2].astype(np.float32) + 1j * raw[1::2].astype(np.float32)
            return iq
        except Exception as e:
            print(f"❌ Error loading IQ file: {e}")
            return None


class WaterfallGenerator:
    """Generates waterfall spectrograms from IQ data"""
    
    def __init__(self, config: RFConfig, colormap: LinearSegmentedColormap):
        self.config = config
        self.colormap = colormap
        
    def generate(self, iq_data: np.ndarray, center_freq: int, output_file: str) -> bool:
        """Generate waterfall plot from IQ data"""
        try:
            overlap = int(self.config.fft_size * 0.75)
            hop = self.config.fft_size - overlap
            window = np.hanning(self.config.fft_size)
            
            frames = (len(iq_data) - self.config.fft_size) // hop
            
            if frames <= 0:
                print("⚠ Not enough samples for waterfall")
                return False
                
            wf = np.empty((frames, self.config.fft_size), dtype=np.float32)
            
            # Compute FFT for each frame
            for i in range(frames):
                s = i * hop
                segment = iq_data[s:s+self.config.fft_size] * window
                spectrum = np.fft.fftshift(np.fft.fft(segment))
                power = 20 * np.log10(np.abs(spectrum) + 1e-12)
                wf[i, :] = power
            
            # Apply fixed dynamic range
            wf_max = np.max(wf)
            wf_min = wf_max - self.config.dynamic_range_db
            wf = np.clip(wf, wf_min, wf_max)
            
            # Generate frequency axis
            center_mhz = center_freq / 1e6
            span_mhz = self.config.sample_rate / 1e6
            x_min = center_mhz - span_mhz / 2
            x_max = center_mhz + span_mhz / 2
            
            # Create and save plot
            plt.figure(figsize=(13, 6))
            plt.imshow(
                wf,
                aspect="auto",
                cmap=self.colormap,
                origin="lower",
                extent=[x_min, x_max, 0, frames]
            )
            plt.colorbar(label="Relative Power (dB)")
            plt.xlabel("Frequency (MHz)")
            plt.ylabel("Time (frames)")
            plt.title(f"Waterfall Spectrum @ {int(center_mhz)} MHz")
            plt.savefig(output_file, dpi=200)
            plt.close()
            
            return True
            
        except Exception as e:
            print(f"❌ Waterfall generation error: {e}")
            return False


class SignalDetection:
    """Represents a detected RF signal"""
    
    def __init__(self, frequency: int, peak_power: float, timestamp: int):
        self.frequency = frequency
        self.peak_power = peak_power
        self.timestamp = timestamp
        self.freq_mhz = int(frequency / 1_000_000)
        self.iq_filename = f"capture_{self.freq_mhz}MHz_{timestamp}.iq"
        self.waterfall_filename = f"waterfall_{self.freq_mhz}MHz_{timestamp}.png"
        
    def __str__(self):
        return f"Signal @ {self.freq_mhz} MHz (Peak: {self.peak_power:.1f} dB)"


class RFMonitor:
    """Main RF monitoring system"""
    
    def __init__(self, config: Optional[RFConfig] = None):
        self.config = config or RFConfig()
        self.sweep = HackRFSweep(self.config)
        self.capture = IQCapture(self.config)
        self.waterfall = WaterfallGenerator(
            self.config, 
            ColorMapGenerator.create_sdr_colormap()
        )
        self.running = False
        self.last_detection: Optional[SignalDetection] = None
        
    def process_detection(self, detection: SignalDetection) -> bool:
        """Process a signal detection: capture IQ and generate waterfall"""
        print(f"\n🔥 {detection}")
        print(f"🎙 Capturing {self.config.capture_seconds}s IQ → {detection.iq_filename}")
        
        # Stop sweep before capture
        self.sweep.stop()
        
        # Capture IQ data
        if not self.capture.capture(detection.frequency, detection.iq_filename):
            print("❌ ERROR: No IQ file written")
            return False
            
        print("📁 IQ Saved. Processing waterfall...")
        
        # Load IQ data
        iq_data = self.capture.load_iq_file(detection.iq_filename)
        if iq_data is None:
            return False
            
        # Generate waterfall
        if self.waterfall.generate(iq_data, detection.frequency, detection.waterfall_filename):
            print(f"🌊 Waterfall Saved → {detection.waterfall_filename}\n")
            return True
        else:
            return False
    
    def scan_once(self) -> Optional[SignalDetection]:
        """Perform one sweep scan and return first detection"""
        print("\n📡 RF Monitoring Active...")
        print("🔍 Sweeping for signals...\n")
        
        process = self.sweep.start()
        
        try:
            while True:
                line = process.stdout.readline().decode().strip()
                parsed = self.sweep.parse_sweep_line(line)
                
                if parsed is None:
                    continue
                    
                freq_start, bin_width, powers = parsed
                peak = max(powers)
                
                # Check if signal exceeds threshold
                if peak > self.config.threshold:
                    idx = powers.index(peak)
                    detected_freq = freq_start + idx * bin_width
                    
                    # Round to nearest MHz for stable tuning
                    detected_freq = int(round(detected_freq / 1e6)) * 1_000_000
                    
                    timestamp = int(time.time())
                    detection = SignalDetection(detected_freq, peak, timestamp)
                    
                    self.sweep.stop()
                    return detection
                    
        except KeyboardInterrupt:
            self.sweep.stop()
            print("\n⚠ Scan interrupted by user")
            return None
    
    def run_single_detection(self) -> Optional[SignalDetection]:
        """Run monitor until one signal is detected and processed"""
        detection = self.scan_once()
        
        if detection:
            self.last_detection = detection
            success = self.process_detection(detection)
            
            if success:
                print("✅ Detection complete. Ready for next step.")
                print(f"📸 Waterfall: {detection.waterfall_filename}")
                print(f"📊 IQ Data: {detection.iq_filename}")
                return detection
            else:
                print("❌ Detection processing failed")
                return None
        else:
            print("⚠ No signal detected")
            return None
    
    def wait_for_continue(self):
        """Wait for user input to continue"""
        input("\n⏸  Press ENTER to scan for next signal (or Ctrl+C to exit)...")


# ============================================================
# MAIN EXECUTION
# ============================================================

def main():
    """Main execution function"""
    
    # Initialize RF Monitor
    monitor = RFMonitor()
    
    print("=" * 60)
    print("  RF SIGNAL DETECTOR - Single Shot Mode")
    print("=" * 60)
    print(f"  Threshold: {monitor.config.threshold} dB")
    print(f"  Frequency Range: {monitor.config.sweep_range} MHz")
    print(f"  Capture Duration: {monitor.config.capture_seconds}s")
    print("=" * 60)
    
    try:
        while True:
            # Run single detection cycle
            detection = monitor.run_single_detection()
            
            if detection:
                # TODO: Call your moondream.py here
                # Example: analyze_with_moondream(detection.waterfall_filename)
                
                # Wait for user to continue
                monitor.wait_for_continue()
            else:
                print("\n🔁 Restarting scan...")
                time.sleep(1)
                
    except KeyboardInterrupt:
        print("\n\n👋 RF Monitor stopped by user")
        monitor.sweep.stop()


