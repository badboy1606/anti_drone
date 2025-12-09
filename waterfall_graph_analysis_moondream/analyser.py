import requests
import base64
import json
import re
from pathlib import Path
from datetime import datetime
from PIL import Image
import io
import numpy as np


# ============================================
#  MOONDREAM CLIENT
# ============================================

class MoondreamAPI:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://api.moondream.ai/v1"
        self.headers = {
            "Content-Type": "application/json",
            "X-Moondream-Auth": api_key
        }

    def _encode_image(self, image_path):
        """Convert image to base64 format supported by Moondream"""
        try:
            with open(image_path, "rb") as f:
                img = base64.b64encode(f.read()).decode()
            ext = Path(image_path).suffix.lower()
            mime = "image/jpeg" if ext in ['.jpg', '.jpeg'] else "image/png"
            return f"data:{mime};base64,{img}"
        except:
            raise Exception(f"❌ Could not load: {image_path}")
    
    def _encode_image_from_pil(self, pil_image):
        """Convert PIL image to base64 format"""
        buffered = io.BytesIO()
        pil_image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return f"data:image/png;base64,{img_str}"
    
    def ask(self, image_input, question):
        """Ask vision question to Moondream
        image_input can be either a file path (str) or PIL Image object
        """
        if isinstance(image_input, str):
            image_data = self._encode_image(image_input)
        else:
            image_data = self._encode_image_from_pil(image_input)
            
        payload = {
            "image_url": image_data,
            "question": question
        }

        try:
            res = requests.post(
                f"{self.base_url}/query",
                headers=self.headers,
                json=payload,
                timeout=25
            )

            if res.status_code == 200:
                return res.json().get("answer","NO ANSWER")
            else:
                raise Exception(f"API Error {res.status_code}: {res.text}")

        except Exception as e:
            return f"ERROR: {str(e)}"



# ============================================
#  IMAGE ANALYZER WITH PIXEL-BASED INTENSITY
# ============================================

class ImageIntensityAnalyzer:
    """Analyzes actual pixel intensities to measure signal strength"""
    
    @staticmethod
    def compute_intensity(pil_image):
        """
        Compute normalized intensity (0.0 to 1.0) based on yellow/red pixels
        Returns intensity score where:
        - 0.0 = no signals (all green/cyan)
        - 1.0 = maximum signals (lots of bright yellow/red)
        """
        img_array = np.array(pil_image.convert('RGB'))
        
        # Define color thresholds for signal detection
        # Yellow: high R, high G, low B
        # Red: high R, low G, low B
        # Green/Cyan background: low R, high G, high B
        
        r = img_array[:, :, 0].astype(float)
        g = img_array[:, :, 1].astype(float)
        b = img_array[:, :, 2].astype(float)
        
        # Detect yellow pixels (R>150, G>150, B<100)
        yellow_mask = (r > 150) & (g > 150) & (b < 100)
        
        # Detect red pixels (R>150, G<100, B<100)
        red_mask = (r > 150) & (g < 100) & (b < 100)
        
        # Detect orange pixels (R>150, G>100, G<200, B<100)
        orange_mask = (r > 150) & (g > 100) & (g < 200) & (b < 100)
        
        # Combine all signal pixels
        signal_mask = yellow_mask | red_mask | orange_mask
        
        # Calculate percentage of signal pixels
        total_pixels = img_array.shape[0] * img_array.shape[1]
        signal_pixels = np.sum(signal_mask)
        signal_ratio = signal_pixels / total_pixels
        
        # Calculate average brightness of signal pixels
        if signal_pixels > 0:
            signal_brightness = np.mean(r[signal_mask]) / 255.0
        else:
            signal_brightness = 0.0
        
        # Combine ratio and brightness for final intensity score
        # Weight: 70% coverage, 30% brightness
        intensity = (signal_ratio * 0.7) + (signal_brightness * 0.3)
        
        # Normalize to 0-1 range (cap at 1.0)
        intensity = min(intensity * 2.5, 1.0)  # Scale factor to use full range
        
        return round(intensity, 3)



# ============================================
#  IMAGE SPLITTER
# ============================================

class ImageSplitter:
    @staticmethod
    def split_vertical(image_path):
        """Split image into left and right halves vertically"""
        try:
            img = Image.open(image_path)
            width, height = img.size
            mid_width = width // 2
            
            # Left half (lower frequencies)
            left_half = img.crop((0, 0, mid_width, height))
            
            # Right half (higher frequencies)
            right_half = img.crop((mid_width, 0, width, height))
            
            return {
                "left": left_half,
                "right": right_half
            }
        except Exception as e:
            raise Exception(f"❌ Could not split image: {str(e)}")
    
    @staticmethod
    def split_horizontal(image_path):
        """Split image into top and bottom halves horizontally (for time analysis)"""
        try:
            img = Image.open(image_path)
            width, height = img.size
            mid_height = height // 2
            
            # Bottom half (earlier time - OLDER signals)
            bottom_half = img.crop((0, mid_height, width, height))
            
            # Top half (recent time - NEWER signals)
            top_half = img.crop((0, 0, width, mid_height))
            
            return {
                "bottom": bottom_half,  # Earlier/older time
                "top": top_half         # Recent/newer time
            }
        except Exception as e:
            raise Exception(f"❌ Could not split image: {str(e)}")



# ============================================
#  DRONE SPECTRUM ANALYZER
# ============================================

class DroneSpectrumAnalyzer:
    
    def __init__(self, api_key):
        self.ai = MoondreamAPI(api_key)
        self.splitter = ImageSplitter()
        self.intensity_analyzer = ImageIntensityAnalyzer()

        # Base prompts for full image analysis
        self.base_prompts = {
            "frequency": "Read the X-axis frequency labels at the bottom. Give only the start and end frequency values in MHz.",
            "overall_signals": "How many distinct horizontal yellow or red signal bands do you see? Answer with only a single number."
        }
        
        # Prompts for split image analysis
        self.split_prompts = {
            "has_signals": "Are there any yellow, orange, or red horizontal lines visible? Answer only: YES or NO",
            "density": "How many horizontal yellow/red/orange signal lines can you count? Answer with only a number (0 if none)."
        }

    def analyze_image(self, image_path):
        print(f"\n🔍 Analyzing spectrum → {image_path}")

        results = {}
        
        # 1. Analyze full image for basic info
        print("\n📊 Analyzing full image...")
        for key, q in self.base_prompts.items():
            ans = self.ai.ask(image_path, q)
            print(f"🧠 {key}: {ans}")
            results[key] = ans
        
        # 2. Split image VERTICALLY and analyze with pixel-based intensity
        print("\n✂ Splitting image vertically (left vs right frequencies)...")
        v_split = self.splitter.split_vertical(image_path)
        
        results["left_half"] = {}
        results["right_half"] = {}
        
        # Analyze left half
        print("\n⬅  Analyzing LEFT half (lower frequencies)...")
        for key, q in self.split_prompts.items():
            ans = self.ai.ask(v_split["left"], q)
            print(f"🧠 left_{key}: {ans}")
            results["left_half"][key] = ans
        
        # Compute pixel-based intensity for left
        left_intensity = self.intensity_analyzer.compute_intensity(v_split["left"])
        results["left_half"]["pixel_intensity"] = left_intensity
        print(f"📊 left_pixel_intensity: {left_intensity}")
        
        # Analyze right half
        print("\n➡  Analyzing RIGHT half (higher frequencies)...")
        for key, q in self.split_prompts.items():
            ans = self.ai.ask(v_split["right"], q)
            print(f"🧠 right_{key}: {ans}")
            results["right_half"][key] = ans
        
        # Compute pixel-based intensity for right
        right_intensity = self.intensity_analyzer.compute_intensity(v_split["right"])
        results["right_half"]["pixel_intensity"] = right_intensity
        print(f"📊 right_pixel_intensity: {right_intensity}")
        
        # 3. Split HORIZONTALLY for temporal analysis
        print("\n✂ Splitting image horizontally (bottom vs top time)...")
        h_split = self.splitter.split_horizontal(image_path)
        
        results["bottom_half"] = {}
        results["top_half"] = {}
        
        # Analyze bottom half (EARLIER/OLDER time)
        print("\n⬇  Analyzing BOTTOM half (EARLIER/OLDER time)...")
        for key, q in self.split_prompts.items():
            ans = self.ai.ask(h_split["bottom"], q)
            print(f"🧠 bottom_{key}: {ans}")
            results["bottom_half"][key] = ans
        
        # Compute pixel-based intensity for bottom
        bottom_intensity = self.intensity_analyzer.compute_intensity(h_split["bottom"])
        results["bottom_half"]["pixel_intensity"] = bottom_intensity
        print(f"📊 bottom_pixel_intensity: {bottom_intensity}")
        
        # Analyze top half (RECENT/NEWER time)
        print("\n⬆  Analyzing TOP half (RECENT/NEWER time)...")
        for key, q in self.split_prompts.items():
            ans = self.ai.ask(h_split["top"], q)
            print(f"🧠 top_{key}: {ans}")
            results["top_half"][key] = ans
        
        # Compute pixel-based intensity for top
        top_intensity = self.intensity_analyzer.compute_intensity(h_split["top"])
        results["top_half"]["pixel_intensity"] = top_intensity
        print(f"📊 top_pixel_intensity: {top_intensity}")

        return self._interpret(results)


    # ---- INTERPRETATION LAYER ----
    def _interpret(self, data):

        # Extract frequency range
        freq_values = re.findall(r"\d+\.?\d*", data.get("frequency",""))
        freq_start = float(freq_values[0]) if freq_values else None
        freq_end = float(freq_values[1]) if len(freq_values)>1 else None

        # Use PIXEL-BASED intensity measurements (0.0 to 1.0)
        left_intensity = data["left_half"]["pixel_intensity"]
        right_intensity = data["right_half"]["pixel_intensity"]
        bottom_intensity = data["bottom_half"]["pixel_intensity"]
        top_intensity = data["top_half"]["pixel_intensity"]
        
        # Extract density counts from AI
        left_density = self._extract_number(data["left_half"]["density"])
        right_density = self._extract_number(data["right_half"]["density"])
        bottom_density = self._extract_number(data["bottom_half"]["density"])
        top_density = self._extract_number(data["top_half"]["density"])
        
        # Frequency analysis
        freq_intensity_diff = right_intensity - left_intensity
        
        if right_intensity > left_intensity + 0.15:
            freq_distribution = "Signals concentrated at HIGHER frequencies (right side)"
            freq_bias = "RIGHT (higher freq)"
        elif left_intensity > right_intensity + 0.15:
            freq_distribution = "Signals concentrated at LOWER frequencies (left side)"
            freq_bias = "LEFT (lower freq)"
        else:
            freq_distribution = "Signals distributed across frequency range"
            freq_bias = "BALANCED"
        
        # TEMPORAL ANALYSIS - CRITICAL FIX
        # Bottom = EARLIER time (past)
        # Top = RECENT time (present)
        # If bottom > top: signals decreasing → drone moving AWAY
        # If top > bottom: signals increasing → drone APPROACHING
        
        intensity_diff = top_intensity - bottom_intensity  # Recent minus Past
        density_diff = top_density - bottom_density
        
        # Combined trend score (positive = approaching, negative = moving away)
        trend_score = (intensity_diff * 0.7) + (density_diff * 0.03)
        
        # Determine temporal state
        if trend_score > 0.15:  # Recent signals STRONGER than past
            temporal_state = "INCREASING ACTIVITY (drone approaching)"
            temp_description = f"Signals stronger in recent time (top: {top_intensity:.3f}) vs earlier (bottom: {bottom_intensity:.3f})"
        elif trend_score < -0.15:  # Recent signals WEAKER than past
            temporal_state = "DECREASING ACTIVITY (drone moving away)"
            temp_description = f"Signals weaker in recent time (top: {top_intensity:.3f}) vs earlier (bottom: {bottom_intensity:.3f})"
        elif max(top_intensity, bottom_intensity) > 0.3:  # Strong signals but stable
            temporal_state = "STABLE ACTIVITY (drone hovering)"
            temp_description = f"Similar signal levels across time (top: {top_intensity:.3f}, bottom: {bottom_intensity:.3f})"
        else:
            temporal_state = "NO SIGNIFICANT ACTIVITY"
            temp_description = f"Minimal signals detected (top: {top_intensity:.3f}, bottom: {bottom_intensity:.3f})"

        # Overall metrics
        total_signals = self._extract_number(data["overall_signals"])
        avg_intensity = (left_intensity + right_intensity + bottom_intensity + top_intensity) / 4
        max_intensity = max(left_intensity, right_intensity, bottom_intensity, top_intensity)

        confidence = self._compute_confidence(
            temporal_state, total_signals, avg_intensity, trend_score,
            right_intensity, left_intensity, max_intensity
        )

        return {
            "frequency_start": freq_start,
            "frequency_end": freq_end,
            "bandwidth": (freq_end - freq_start) if freq_start and freq_end else None,
            "total_signal_count": total_signals,
            "average_signal_intensity": round(avg_intensity, 3),
            "max_signal_intensity": round(max_intensity, 3),
            
            "frequency_analysis": {
                "distribution": freq_distribution,
                "left_density": left_density,
                "left_intensity": round(left_intensity, 3),
                "right_density": right_density,
                "right_intensity": round(right_intensity, 3),
                "freq_bias": freq_bias,
                "intensity_difference": round(freq_intensity_diff, 3)
            },
            
            "temporal_analysis": {
                "trend": temporal_state,
                "description": temp_description,
                "trend_score": round(trend_score, 3),
                "bottom_density": bottom_density,
                "bottom_intensity": round(bottom_intensity, 3),
                "top_density": top_density,
                "top_intensity": round(top_intensity, 3),
                "intensity_change": round(intensity_diff, 3)
            },
            
            "confidence": confidence,
            
            "raw_responses": {
                "left": data["left_half"]["has_signals"],
                "right": data["right_half"]["has_signals"],
                "bottom": data["bottom_half"]["has_signals"],
                "top": data["top_half"]["has_signals"]
            }
        }

    def _extract_number(self, value):
        """Extract first number from response"""
        numbers = re.findall(r"\d+", str(value))
        if numbers:
            return int(numbers[0])
        value_lower = str(value).lower()
        if "yes" in value_lower and "no" not in value_lower:
            return 5
        return 0

    def _compute_confidence(self, trend, total_bands, avg_intensity, trend_score,
                           right_intensity, left_intensity, max_intensity):
        score = 0

        # Temporal trend confidence
        if "approaching" in trend.lower():
            score += 30
        elif "moving away" in trend.lower():
            score += 30  # Moving away is just as confident as approaching
        elif "hovering" in trend.lower():
            score += 20

        # Signal presence based on actual intensity
        score += min(int(avg_intensity * 100), 30)
        
        # Strong signals boost
        if max_intensity > 0.5:
            score += 15
        elif max_intensity > 0.3:
            score += 10
        
        # Clear trend boost
        if abs(trend_score) > 0.25:
            score += 15
        elif abs(trend_score) > 0.15:
            score += 10
        
        # Clear frequency pattern
        if abs(right_intensity - left_intensity) > 0.2:
            score += 10

        if score > 85: tag = "VERY HIGH"
        elif score > 65: tag = "HIGH"
        elif score > 40: tag = "MEDIUM"
        elif score > 20: tag = "LOW"
        else: tag = "VERY LOW"

        return f"{tag} ({score}%)"



# ============================================
#  TREND OVER TIME (SERIES MODE)
# ============================================

class DroneTrendMonitor:
    
    def __init__(self, api_key):
        self.analyzer = DroneSpectrumAnalyzer(api_key)
        self.history = []

    def run_series(self, images):
        for img in images:
            r = self.analyzer.analyze_image(img)
            self.history.append(r)

        return self._trend()

    def _trend(self):
        if len(self.history) < 2:
            return "Not enough images for comparison."

        # Use intensity changes instead of just counts
        intensity_diffs = [
            self.history[i]["average_signal_intensity"] - self.history[i-1]["average_signal_intensity"]
            for i in range(1, len(self.history))
        ]

        velocity = sum(intensity_diffs)

        if velocity > 0.3: trend = "Rapid Increase"
        elif velocity > 0.1: trend = "Slow Increase"
        elif abs(velocity) <= 0.1: trend = "Stable Pattern"
        elif velocity < -0.3: trend = "Rapid Decline"
        else: trend = "Slow Decline"

        return {
            "history": self.history,
            "pattern_velocity": trend,
            "total_intensity_change": round(velocity, 3)
        }



# ============================================
#  RUN EXAMPLE
# ============================================

if __name__ == "__main__":
    KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJrZXlfaWQiOiJiMjI2M2U4Zi02ODE1LTQ1ZmQtODlkZC05MGQ5OGU2MmRiYTkiLCJvcmdfaWQiOiJxSHdoVlhGSkZDVlZxSkVmVTh3TG0yVXhtczZ5bHA1eSIsImlhdCI6MTc2NTE5ODExNywidmVyIjoxfQ.r1qik5ey-c-0atkYDo62aVRdtz9_qKRiBERA3YGfiVU"

    analyzer = DroneSpectrumAnalyzer(KEY)
    result = analyzer.analyze_image("image1.jpg")

    print("\n" + "="*70)
    print("📌 COMPREHENSIVE RESULT SUMMARY")
    print("="*70)
    print(json.dumps(result, indent=2))