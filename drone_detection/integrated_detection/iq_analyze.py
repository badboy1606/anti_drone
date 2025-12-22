import re
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.stats import kurtosis, skew
from dataclasses import dataclass
import os
from scipy.ndimage import binary_closing
from scipy.fft import fftshift
import json
from datetime import datetime

@dataclass
class DroneSignature:
    name: str
    freq_band: tuple
    bandwidth: tuple
    signal_type: str
    duty_cycle: tuple
    persistence: str
    modulation: str = "Unknown"

# Expanded and refined drone signatures
DRONE_SIGNATURES = [
    DroneSignature("DJI OcuSync 2.0 (2.4 GHz)", (2.4e9, 2.48e9), (10e6, 20e6), "OFDM", (0.75, 1.0), "continuous", "OFDM"),
    DroneSignature("DJI OcuSync 2.0 (5.8 GHz)", (5.725e9, 5.85e9), (10e6, 20e6), "OFDM", (0.75, 1.0), "continuous", "OFDM"),
    DroneSignature("DJI OcuSync 3.0 (2.4 GHz)", (2.4e9, 2.48e9), (15e6, 40e6), "OFDM", (0.8, 1.0), "continuous", "OFDM"),
    DroneSignature("DJI OcuSync 3.0 (5.8 GHz)", (5.725e9, 5.85e9), (15e6, 40e6), "OFDM", (0.8, 1.0), "continuous", "OFDM"),
    DroneSignature("FPV Analog Video (5.8 GHz)", (5.65e9, 5.95e9), (6e6, 10e6), "Analog FM", (0.85, 1.0), "continuous", "FM"),
    DroneSignature("FPV Analog Video (1.2-1.3 GHz)", (1.2e9, 1.36e9), (6e6, 10e6), "Analog FM", (0.85, 1.0), "continuous", "FM"),
    DroneSignature("DJI Lightbridge (2.4 GHz)", (2.4e9, 2.483e9), (8e6, 20e6), "OFDM", (0.7, 1.0), "continuous", "OFDM"),
    DroneSignature("Long-Range RC (900 MHz)", (902e6, 928e6), (100e3, 500e3), "FSK/Telemetry", (0.1, 0.4), "burst", "FSK"),
    DroneSignature("ExpressLRS (2.4 GHz)", (2.4e9, 2.48e9), (2e6, 6e6), "LoRa", (0.3, 0.7), "burst", "LoRa"),
    DroneSignature("Crossfire (900 MHz)", (868e6, 928e6), (200e3, 600e3), "Telemetry", (0.2, 0.5), "burst", "Proprietary"),
    DroneSignature("WiFi Drones (2.4 GHz)", (2.412e9, 2.472e9), (20e6, 40e6), "WiFi OFDM", (0.3, 0.8), "mixed", "WiFi"),
    DroneSignature("WiFi Drones (5 GHz)", (5.15e9, 5.85e9), (20e6, 80e6), "WiFi OFDM", (0.3, 0.8), "mixed", "WiFi"),
    # Add more flexible generic signatures
    DroneSignature("Generic 2.4GHz Control", (2.3e9, 2.5e9), (500e3, 50e6), "Various", (0.2, 1.0), "mixed", "Various"),
    DroneSignature("Generic 5.8GHz Link", (5.6e9, 5.95e9), (3e6, 50e6), "Various", (0.3, 1.0), "mixed", "Various"),
]

INTERFERENCE_SIGNATURES = {
    'WiFi_2.4GHz': {'freq': (2.412e9, 2.472e9), 'bw': (20e6, 40e6), 'type': 'bursty', 'typical_cv': 0.8},
    'WiFi_5GHz': {'freq': (5.15e9, 5.85e9), 'bw': (20e6, 160e6), 'type': 'bursty', 'typical_cv': 0.7},
    'Bluetooth': {'freq': (2.402e9, 2.480e9), 'bw': (1e6, 2e6), 'type': 'hopping', 'typical_cv': 1.2},
    'Microwave': {'freq': (2.45e9, 2.46e9), 'bw': (50e6, 100e6), 'type': 'continuous', 'typical_cv': 0.1},
}

def extract_freq_from_filename(path):
    """Extract frequency from filename with multiple pattern matching"""
    name = os.path.basename(path).lower()
    # Pattern 1: 2.4ghz, 5.8ghz, etc.
    m = re.search(r'(\d+(?:\.\d+)?)(?:\s?)(ghz|mhz|khz|hz)', name)
    if m:
        num = float(m.group(1))
        unit = m.group(2)
        if 'ghz' in unit: num *= 1e9
        elif 'mhz' in unit: num *= 1e6
        elif 'khz' in unit: num *= 1e3
        return int(num)
    # Pattern 2: plain integer like 2400000000
    m2 = re.search(r'([1-9]\d{8,9})', name)
    if m2:
        return int(m2.group(1))
    return None

class DroneDetector:
    def __init__(self, filepath, sample_rate=1e6, center_freq=None, data_type='complex64', 
                 max_samples=None, debug=True):
        """
        Initialize drone detector
        
        Args:
            filepath: Path to IQ file
            sample_rate: Sample rate in Hz (default: 1 MHz)
            center_freq: Center frequency in Hz (if None, attempts to extract from filename)
            data_type: Data type ('complex64', 'complex128', 'int16', 'int8', 'float32')
            max_samples: Maximum samples to load (None = all)
            debug: Enable debug output
        """
        self.filepath = filepath
        self.sample_rate = float(sample_rate)
        self.center_freq = center_freq
        self.data_type = data_type
        self.max_samples = max_samples
        self.debug = debug
        self.iq_data = None
        self.detection_results = {}
        self.metadata = {
            'timestamp': datetime.now().isoformat(),
            'filename': os.path.basename(filepath)
        }
        
    def load_iq_file(self):
        """Load IQ data from file with support for multiple formats"""
        print(f"Loading IQ file: {self.filepath}")
        
        try:
            if self.data_type == 'complex64':
                data = np.fromfile(self.filepath, dtype=np.complex64, count=self.max_samples or -1)
            elif self.data_type == 'complex128':
                data = np.fromfile(self.filepath, dtype=np.complex128, count=self.max_samples or -1)
            elif self.data_type == 'int16':
                count = (self.max_samples * 2) if self.max_samples else -1
                raw = np.fromfile(self.filepath, dtype=np.int16, count=count)
                data = (raw[::2] + 1j * raw[1::2]) / 32768.0
            elif self.data_type == 'int8':
                count = (self.max_samples * 2) if self.max_samples else -1
                raw = np.fromfile(self.filepath, dtype=np.int8, count=count)
                data = (raw[::2] + 1j * raw[1::2]) / 128.0
            elif self.data_type == 'float32':
                count = (self.max_samples * 2) if self.max_samples else -1
                raw = np.fromfile(self.filepath, dtype=np.float32, count=count)
                data = raw[::2] + 1j * raw[1::2]
            else:
                raise ValueError(f"Unsupported data_type: {self.data_type}")

            if data is None or len(data) == 0:
                raise ValueError("No data loaded from file.")

            self.iq_data = data
            duration = len(self.iq_data) / self.sample_rate
            file_size_mb = os.path.getsize(self.filepath) / (1024 * 1024)
            
            # Check signal power
            avg_power = np.mean(np.abs(self.iq_data)**2)
            max_power = np.max(np.abs(self.iq_data)**2)
            
            print(f"✓ Loaded {len(self.iq_data):,} samples ({duration:.3f} seconds, {file_size_mb:.2f} MB)")
            if self.debug:
                print(f"   Signal power - Avg: {10*np.log10(avg_power+1e-14):.1f} dB, Max: {10*np.log10(max_power+1e-14):.1f} dB")
            
            self.metadata.update({
                'num_samples': len(self.iq_data),
                'duration_seconds': duration,
                'file_size_mb': file_size_mb,
                'avg_power_db': 10*np.log10(avg_power+1e-14),
                'max_power_db': 10*np.log10(max_power+1e-14)
            })
            
        except Exception as e:
            print(f"❌ Error loading file: {e}")
            raise

    def calculate_psd(self, nfft=4096):
        """Calculate Power Spectral Density using Welch's method"""
        print("   Computing PSD using Welch's method...")
        data = self.iq_data
        
        # Adaptive decimation for large files
        if len(data) > 10_000_000:
            decimate_factor = max(1, int(len(data) / 5_000_000))
            data = data[::decimate_factor]
            if self.debug:
                print(f"   Decimated to {len(data):,} samples (factor {decimate_factor})")

        # Adaptive segment sizing - REDUCED for better frequency resolution
        nperseg = min(nfft, max(1024, len(data)//4))
        noverlap = int(nperseg * 0.75)  # Increased overlap
        
        f, psd = signal.welch(data, fs=self.sample_rate, nperseg=nperseg, 
                              noverlap=noverlap, return_onesided=False, 
                              scaling='density', window='hann')
        
        f = fftshift(f)
        psd = fftshift(psd)
        psd_db = 10 * np.log10(psd + 1e-14)
        
        # Convert to absolute frequency if center_freq known
        if self.center_freq is not None:
            freq_shifted = f + self.center_freq
        else:
            freq_shifted = f
        
        if self.debug:
            print(f"   PSD computed: {len(psd_db)} frequency bins, Resolution: {(f[1]-f[0])/1e3:.1f} kHz")
            
        return freq_shifted, psd_db, f

    def analyze_signal_features(self, region_data):
        """Extract comprehensive signal features for classification"""
        if region_data is None or len(region_data) < 10:
            return {}
            
        # Power-based features
        power = np.abs(region_data)**2
        mean_power = np.mean(power)
        std_power = np.std(power)
        cv_power = std_power / (mean_power + 1e-14)
        
        # Statistical features
        kurt = kurtosis(power)
        skewness = skew(power)
        
        # Spectral entropy
        power_norm = power / (np.sum(power) + 1e-14)
        spectral_entropy = -np.sum(power_norm * np.log2(power_norm + 1e-14))
        
        # Temporal stability analysis
        chunk_size = max(100, len(region_data)//20)
        n_chunks = max(1, len(region_data) // chunk_size)
        chunk_powers = []
        for i in range(n_chunks):
            chunk = region_data[i*chunk_size:(i+1)*chunk_size]
            if len(chunk) > 0:
                chunk_powers.append(np.mean(np.abs(chunk)**2))
        
        if len(chunk_powers) > 1:
            temporal_stability = 1.0 - (np.std(chunk_powers) / (np.mean(chunk_powers) + 1e-14))
        else:
            temporal_stability = 0.5  # Neutral if can't compute
        
        # Phase features
        try:
            instantaneous_phase = np.angle(region_data)
            phase_diff = np.diff(instantaneous_phase)
            phase_diff = np.unwrap(phase_diff)
            phase_stability = 1.0 / (np.std(phase_diff) + 1e-14)
        except:
            phase_stability = 0.0
        
        # Amplitude features
        amplitude = np.abs(region_data)
        amplitude_crest_factor = np.max(amplitude) / (np.mean(amplitude) + 1e-14)
        
        return {
            'cv_power': cv_power,
            'kurtosis': kurt,
            'skewness': skewness,
            'spectral_entropy': spectral_entropy,
            'temporal_stability': temporal_stability,
            'mean_power': mean_power,
            'phase_stability': phase_stability,
            'amplitude_crest_factor': amplitude_crest_factor
        }

    def detect_active_bands(self, psd_db, freq_shifted, freq_baseband, 
                           min_bandwidth=500e3, min_peak_db_above_noise=4.0):
        """Detect active frequency bands with RELAXED thresholding for better sensitivity"""
        # More aggressive noise floor estimation
        noise_samples = np.concatenate([psd_db[:len(psd_db)//10], psd_db[-len(psd_db)//10:]])
        noise_floor = np.percentile(noise_samples, 25)
        
        psd_max = np.max(psd_db)
        dynamic_range = psd_max - noise_floor
        
        # RELAXED threshold - reduced from 0.15 to 0.10
        threshold = noise_floor + max(min_peak_db_above_noise, dynamic_range * 0.10)
        
        print(f"   Noise floor: {noise_floor:.1f} dB, Max: {psd_max:.1f} dB, Dynamic range: {dynamic_range:.1f} dB")
        print(f"   Detection threshold: {threshold:.1f} dB (RELAXED for better sensitivity)")
        
        # Binary mask with morphological closing to merge nearby signals
        active_mask = psd_db > threshold
        # Increased structure size to merge signals better
        active_mask = binary_closing(active_mask, structure=np.ones(11))
        
        if self.debug:
            print(f"   Active bins: {np.sum(active_mask)} / {len(active_mask)} ({100*np.sum(active_mask)/len(active_mask):.1f}%)")
        
        # Extract regions
        active_regions = []
        in_region = False
        start_idx = 0
        
        for i, active in enumerate(active_mask):
            if active and not in_region:
                start_idx = i
                in_region = True
            elif not active and in_region:
                end_idx = i - 1
                self._process_region(start_idx, end_idx, freq_shifted, psd_db, 
                                   noise_floor, min_bandwidth, min_peak_db_above_noise, 
                                   active_regions)
                in_region = False
        
        # Handle region at end
        if in_region:
            end_idx = len(active_mask) - 1
            self._process_region(start_idx, end_idx, freq_shifted, psd_db, 
                               noise_floor, min_bandwidth, min_peak_db_above_noise, 
                               active_regions)
        
        return active_regions, noise_floor, threshold

    def _process_region(self, start_idx, end_idx, freq_shifted, psd_db, 
                       noise_floor, min_bandwidth, min_peak_db_above_noise, active_regions):
        """Helper to process a detected region"""
        bw = abs(freq_shifted[end_idx] - freq_shifted[start_idx])
        peak_db = np.max(psd_db[start_idx:end_idx+1])
        avg_db = np.mean(psd_db[start_idx:end_idx+1])
        
        if bw >= min_bandwidth and (peak_db - noise_floor) >= min_peak_db_above_noise:
            freq_range = (freq_shifted[start_idx], freq_shifted[end_idx])
            region_iq = self.extract_frequency_band(freq_range)
            features = self.analyze_signal_features(region_iq) if region_iq is not None else {}
            
            active_regions.append({
                'start_idx': start_idx,
                'end_idx': end_idx,
                'freq_start': freq_shifted[start_idx],
                'freq_end': freq_shifted[end_idx],
                'freq_center': (freq_shifted[start_idx] + freq_shifted[end_idx]) / 2,
                'bandwidth': bw,
                'power_db': avg_db,
                'peak_power_db': peak_db,
                'snr_db': peak_db - noise_floor,
                'features': features
            })
            
            if self.debug:
                print(f"   Region: {freq_range[0]/1e9:.4f}-{freq_range[1]/1e9:.4f} GHz, BW: {bw/1e6:.2f} MHz, SNR: {peak_db - noise_floor:.1f} dB")

    def extract_frequency_band(self, freq_range):
        """Extract IQ samples for a specific frequency range"""
        try:
            max_samples = min(len(self.iq_data), 100_000)  # Reduced for speed
            data = self.iq_data[:max_samples]
            
            if self.center_freq is None:
                return data
            
            nyq = self.sample_rate / 2.0
            low = (freq_range[0] - self.center_freq) / nyq
            high = (freq_range[1] - self.center_freq) / nyq
            
            # Clamp to valid range
            low = np.clip(low, -0.999, 0.999)
            high = np.clip(high, -0.999, 0.999)
            
            if abs(high - low) < 0.01:
                return data  # Return full data if range too small
            
            # Design and apply bandpass filter
            sos = signal.butter(4, [low, high], btype='band', output='sos')
            filtered = signal.sosfilt(sos, data)
            
            return filtered
            
        except Exception as e:
            if self.debug:
                print(f"   Warning: Could not extract frequency band: {e}")
            return self.iq_data[:min(len(self.iq_data), 100_000)]

    def classify_signal_type(self, region):
        """Classify signal as drone/interference/unknown with RELAXED scoring"""
        fc = region['freq_center']
        bw = region['bandwidth']
        features = region.get('features', {})
        snr = region.get('snr_db', 0)
        
        temporal_stability = features.get('temporal_stability', 0.5)
        cv_power = features.get('cv_power', 1.0)
        spectral_entropy = features.get('spectral_entropy', 5.0)
        phase_stability = features.get('phase_stability', 0.0)
        
        reasons = []
        drone_score = 0
        interference_score = 0
        
        # SNR-based confidence
        if snr > 12:
            drone_score += 15
            reasons.append(f"High SNR ({snr:.1f} dB)")
        elif snr > 6:
            drone_score += 8
            reasons.append(f"Good SNR ({snr:.1f} dB)")
        
        # Temporal stability (RELAXED thresholds)
        if temporal_stability > 0.6:
            drone_score += 30
            reasons.append(f"Stable temporal pattern ({temporal_stability:.2f})")
        elif temporal_stability > 0.4:
            drone_score += 15
            reasons.append(f"Moderate temporal stability ({temporal_stability:.2f})")
        elif temporal_stability < 0.25:
            interference_score += 20
            reasons.append(f"Unstable/bursty pattern ({temporal_stability:.2f})")
        
        # Power coefficient of variation (RELAXED)
        if cv_power < 0.5:
            drone_score += 20
            reasons.append(f"Consistent power ({cv_power:.2f})")
        elif cv_power > 1.2:
            interference_score += 15
            reasons.append(f"Highly variable power ({cv_power:.2f})")
        
        # Spectral entropy (RELAXED)
        if 2.0 <= spectral_entropy <= 7.0:
            drone_score += 12
            reasons.append(f"Typical drone entropy ({spectral_entropy:.2f})")
        elif spectral_entropy > 8.5:
            interference_score += 12
            reasons.append(f"High entropy - possibly WiFi ({spectral_entropy:.2f})")
        
        # Phase stability
        if phase_stability > 30:
            drone_score += 8
            reasons.append("Stable phase")
        
        # Frequency-specific heuristics (MORE GENEROUS)
        if self.center_freq is not None:
            if 1.15e9 <= fc <= 1.4e9:
                drone_score += 40
                reasons.append("1.2 GHz FPV band")
            elif 5.6e9 <= fc <= 5.96e9:
                if 5e6 <= bw <= 15e6:
                    drone_score += 35
                    reasons.append("5.8 GHz FPV video band")
                elif bw > 15e6:
                    drone_score += 30
                    reasons.append("5.8 GHz digital link")
                else:
                    drone_score += 20
                    reasons.append("5.8 GHz drone band")
            elif 2.3e9 <= fc <= 2.5e9:
                if 5e6 <= bw <= 50e6:
                    drone_score += 25
                    reasons.append("2.4 GHz drone bandwidth range")
                else:
                    drone_score += 10
                    reasons.append("2.4 GHz ISM band")
            elif 850e6 <= fc <= 950e6:
                if bw < 2e6:
                    drone_score += 35
                    reasons.append("900 MHz RC telemetry band")
        
        # Bandwidth-specific patterns (RELAXED)
        if 5e6 <= bw <= 12e6:
            drone_score += 12
            reasons.append("Typical FPV video bandwidth")
        elif 10e6 <= bw <= 50e6:
            drone_score += 10
            reasons.append("Typical digital drone link bandwidth")
        
        # RELAXED decision logic - reduced thresholds from 70 to 55
        if drone_score >= 55 and drone_score > interference_score + 10:
            return 'drone', min(100, drone_score), reasons
        elif interference_score >= 65 and interference_score > drone_score + 15:
            return 'interference', min(100, interference_score), reasons
        else:
            # Give benefit of doubt - if any drone indicators, lean towards drone
            if drone_score > 40:
                return 'drone', min(100, drone_score), reasons + ["(Benefit of doubt)"]
            return 'unknown', min(100, max(drone_score, interference_score)), reasons

    def match_drone_signatures(self, active_regions):
        """Match detected signals to known drone signatures with RELAXED matching"""
        matches = []
        filtered_out = []
        
        for region in active_regions:
            fc = region['freq_center']
            bw = region['bandwidth']
            
            # Classify signal
            signal_type, type_conf, reasons = self.classify_signal_type(region)
            
            if self.debug:
                print(f"   Classifying region @ {fc/1e9:.4f} GHz: {signal_type} ({type_conf}%)")
            
            # RELAXED filtering - only filter very strong interference
            if signal_type == 'interference' and type_conf >= 85:
                filtered_out.append({
                    'region': region,
                    'reason': 'Strong interference signature',
                    'confidence': type_conf,
                    'details': reasons
                })
                continue
            
            # Match against signatures
            best_match = None
            best_score = 0
            
            for sig in DRONE_SIGNATURES:
                score = 0
                
                # Frequency matching (RELAXED - wider tolerance)
                if self.center_freq is not None:
                    freq_margin = 100e6  # 100 MHz margin
                    if sig.freq_band[0] - freq_margin <= fc <= sig.freq_band[1] + freq_margin:
                        if sig.freq_band[0] <= fc <= sig.freq_band[1]:
                            score += 40
                        else:
                            score += 25  # Partial credit for nearby
                    else:
                        continue
                else:
                    score += 15  # More credit if center unknown
                
                # Bandwidth matching (RELAXED - wider tolerance)
                bw_min, bw_max = sig.bandwidth
                if bw_min * 0.5 <= bw <= bw_max * 1.5:
                    if bw_min <= bw <= bw_max:
                        score += 30
                    else:
                        score += 18  # Partial credit
                else:
                    continue
                
                # Signal type bonus (INCREASED)
                if signal_type == 'drone':
                    score += type_conf * 0.4
                elif signal_type == 'unknown':
                    score += type_conf * 0.2  # Give some credit for unknowns
                
                # Modulation/persistence matching
                features = region.get('features', {})
                temporal_stability = features.get('temporal_stability', 0.5)
                
                if sig.persistence == 'continuous' and temporal_stability > 0.5:
                    score += 12
                elif sig.persistence == 'burst' and temporal_stability < 0.6:
                    score += 8
                
                if score > best_score:
                    best_score = score
                    best_match = {
                        'signature': sig,
                        'region': region,
                        'confidence': 'HIGH' if score >= 80 else 'MEDIUM' if score >= 55 else 'LOW',
                        'score': int(min(100, score)),
                        'signal_classification': signal_type,
                        'classification_confidence': type_conf,
                        'classification_reasons': reasons
                    }
            
            # RELAXED threshold - reduced from 60 to 45
            if best_match and best_score >= 45:
                matches.append(best_match)
                if self.debug:
                    print(f"   ✓ Matched: {best_match['signature'].name} ({best_score}%)")
        
        matches.sort(key=lambda x: x['score'], reverse=True)
        return matches, filtered_out

    def analyze_temporal_activity(self):
        """Analyze temporal characteristics of the signal"""
        chunk_size = int(self.sample_rate * 0.05)  # 50ms chunks
        chunk_size = max(chunk_size, 128)
        
        max_samples = min(len(self.iq_data), int(self.sample_rate * 3))
        data = self.iq_data[:max_samples]
        
        n_chunks = max(1, len(data) // chunk_size)
        power_per_chunk = []
        
        for i in range(n_chunks):
            chunk = data[i*chunk_size:(i+1)*chunk_size]
            if len(chunk) > 0:
                power_per_chunk.append(np.mean(np.abs(chunk)**2))
        
        power_per_chunk = np.array(power_per_chunk)
        power_std = np.std(power_per_chunk)
        power_mean = np.mean(power_per_chunk) + 1e-14
        coefficient_of_variation = power_std / power_mean
        
        is_persistent = coefficient_of_variation < 0.5  # Relaxed from 0.4
        is_bursty = coefficient_of_variation > 0.7  # Relaxed from 0.8
        
        return is_persistent, is_bursty, coefficient_of_variation

    def plot_summary(self, freq_shifted, psd_db, active_regions, noise_floor, 
                    threshold, matches, filtered_out):
        """Generate comprehensive visualization"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))
        
        # Main PSD plot
        if self.center_freq is None:
            x = freq_shifted / 1e6
            ax1.set_xlabel('Baseband Frequency (MHz)', fontsize=11)
        else:
            x = freq_shifted / 1e9
            ax1.set_xlabel('Frequency (GHz)', fontsize=11)
        
        ax1.plot(x, psd_db, linewidth=0.7, label='PSD', color='#2E86AB', alpha=0.9)
        ax1.axhline(y=noise_floor, color='gray', linestyle='--', linewidth=1.5, 
                    label=f"Noise Floor: {noise_floor:.1f} dB")
        ax1.axhline(y=threshold, color='red', linestyle='--', linewidth=1.5, 
                    label=f"Threshold: {threshold:.1f} dB")

        # Highlight active regions
        for i, region in enumerate(active_regions):
            xs = (region['freq_start']/1e9, region['freq_end']/1e9)
            ax1.axvspan(xs[0], xs[1], alpha=0.15, color='cyan',
                        label="Active Region" if i == 0 else "")

        # Highlight drone matches
        for i, match in enumerate(matches):
            region = match['region']
            xs = (region['freq_start']/1e9, region['freq_end']/1e9)
            ax1.axvspan(xs[0], xs[1], alpha=0.3, color='red',
                        label="Drone Match" if i == 0 else "")

        # Highlight filtered interference
        for i, filt in enumerate(filtered_out):
            region = filt['region']
            xs = (region['freq_start']/1e9, region['freq_end']/1e9)
            ax1.axvspan(xs[0], xs[1], alpha=0.25, color='orange',
                        label="Interference" if i == 0 else "")

        ax1.set_ylabel("PSD (dB)")
        ax1.set_title("RF Spectrum - Drone Detection Summary")
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc="upper right", fontsize=9)

        # Temporal Power Plot
        ax2.set_title("Temporal Power Distribution")
        chunk_size = int(self.sample_rate * 0.1)
        max_samples = min(len(self.iq_data), int(self.sample_rate * 2))
        data = self.iq_data[:max_samples]

        n_chunks = max(1, len(data)//chunk_size)
        power_vals = []
        for i in range(n_chunks):
            chunk = data[i*chunk_size:(i+1)*chunk_size]
            if len(chunk) > 0:
                power_vals.append(np.mean(np.abs(chunk)**2))

        if len(power_vals) > 0:
            time_axis = np.arange(len(power_vals)) * (chunk_size/self.sample_rate)
            ax2.plot(time_axis, 10*np.log10(np.array(power_vals) + 1e-14),
                     color="#A23B72", linewidth=1.5)

        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Power (dB)")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig("drone_detection_enhanced.png", dpi=150)
        plt.close()
        print("✓ Saved visualization: drone_detection_enhanced.png")

    def print_detection_report(self, matches, filtered_out, active_regions, is_persistent, is_bursty):
        print("\n" + "="*90)
        print(" " * 30 + "DRONE DETECTION REPORT")
        print("="*90)

        print(f"File: {os.path.basename(self.filepath)}")
        print(f"Center Frequency: {self.center_freq/1e9:.3f} GHz")
        print(f"Sample Rate: {self.sample_rate/1e6:.2f} MHz")
        print(f"Duration: {len(self.iq_data)/self.sample_rate:.2f} sec")
        print("="*90)

        print(f"Active RF Regions: {len(active_regions)}")
        print(f"Filtered Interference: {len(filtered_out)}")
        print(f"Potential Drone Matches: {len(matches)}\n")

        if len(matches) > 0:
            print("🚨 DRONE SIGNALS FOUND 🚨\n")
            for i, m in enumerate(matches, 1):
                r = m['region']
                print(f"[{i}] {m['signature'].name}")
                print(f"    Confidence: {m['confidence']} ({m['score']}%)")
                print(f"    Center Freq: {r['freq_center']/1e9:.4f} GHz")
                print(f"    Bandwidth: {r['bandwidth']/1e6:.2f} MHz")
                print(f"    Classification: {m['signal_classification']} ({m['classification_confidence']}%)")
                print("    Evidence:")
                for reason in m['classification_reasons'][:4]:
                    print(f"       - {reason}")
                print()
        else:
            print("✓ No drone-like signals detected.\n")

        print("="*90)
        print("FINAL RESULT:")

        if len(matches) > 0:
            print("🚨 DRONE DETECTED")
        else:
            print("✓ NO DRONE DETECTED")

        print("="*90)

        return len(matches) > 0

    def detect_drones(self):
        print("\n🔍 Starting Drone Detection...\n")

        self.load_iq_file()
        freq_shifted, psd_db, freq_baseband = self.calculate_psd()

        active_regions, noise_floor, threshold = self.detect_active_bands(
            psd_db, freq_shifted, freq_baseband
        )

        matches, filtered_out = self.match_drone_signatures(active_regions)

        is_persistent, is_bursty, cv = self.analyze_temporal_activity()

        self.plot_summary(freq_shifted, psd_db, active_regions, noise_floor,
                          threshold, matches, filtered_out)

        drone_detected = self.print_detection_report(
            matches, filtered_out, active_regions, is_persistent, is_bursty
        )

        self.detection_results = {
            'drone_detected': drone_detected,
            'matches': matches,
            'filtered_out': filtered_out,
            'active_regions': active_regions
        }

        return drone_detected, matches


if __name__ == "__main__":
    file_path ="FLYSKY%20FS%20I6X/FLYSKY FS I6X/pack1_5-6s.iq"
    sample_rate = 20e6
    center_freq = 2.4e9
    data_type = "complex64"
    max_samples = 5_000_000

    print("="*90)
    print("        ADVANCED DRONE DETECTOR (RF ANALYSIS ENGINE)")
    print("="*90)

    detector = DroneDetector(
        filepath=file_path,
        sample_rate=sample_rate,
        center_freq=center_freq,
        data_type=data_type,
        max_samples=max_samples
    )

    drone_detected, results = detector.detect_drones()

    print("\n" + "="*90)
    print("FINAL OUTPUT:")
    print("="*90)

    if drone_detected:
        print("🚨 RESULT: DRONE DETECTED")
    else:
        print("✓ RESULT: NO DRONE DETECTED")

    print("="*90)