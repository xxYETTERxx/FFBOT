import numpy as np
import sounddevice as sd
import queue
import time
import sys
import atexit
import aubio

# Make stdout unbuffered for real-time printing
sys.stdout.reconfigure(line_buffering=True)

class AubioBPMDetector:
    def __init__(self):
        # Audio parameters
        self.sample_rate = 44100
        self.block_size = int(44100)
        self.hop_size = 256  # Standard hop size for aubio
        self.volume_threshold = 0.01
        self.audio_queue = queue.Queue()
        
        # Initialize aubio tempo detection
        self.tempo = aubio.tempo(
            method="phase",
            buf_size=self.hop_size * 4,  # Using larger window for tempo detection
            hop_size=self.hop_size,
            samplerate=self.sample_rate
        )
        
        self.is_running = False
        atexit.register(self.cleanup)
        
    def cleanup(self):
        print("\nCleaning up...")
        self.is_running = False

    def audio_callback(self, indata, frames, time, status):
        if status:
            print(status)
        self.audio_queue.put_nowait(indata.copy())

    def analyze_tempo(self, audio_data):
        """Calculate tempo using aubio"""
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)
        
        # Convert to float32
        audio_mono = audio_data.astype(np.float32)
        
        # Process in chunks
        beats = []
        max_confidence = 0.0
        
        for i in range(0, len(audio_mono) - self.hop_size, self.hop_size):
            chunk = audio_mono[i:i + self.hop_size]
            is_beat = self.tempo(chunk)
            confidence = self.tempo.get_confidence()
            
            if is_beat:
                beats.append(i/self.sample_rate)
            
            max_confidence = max(max_confidence, confidence)
        
        # Calculate BPM from detected beats
        if len(beats) > 1:
            intervals = np.diff(beats)
            bpm = 60 / np.mean(intervals)
        else:
            bpm = self.tempo.get_bpm()
        
        return {
            'bpm': float(bpm),
            'confidence': float(max_confidence),
            'beat_times': beats
        }

    def find_vb_cable(self):
        devices = sd.query_devices()
        for i, device in enumerate(devices):
            if (device['max_input_channels'] > 0 and 
                any(name in device['name'].lower() for name in ['vb', 'cable', 'virtual'])):
                print(f"Found VB-Cable: {device['name']}")
                return i
        print("VB-Cable not found, using default input")
        return None

    def run(self):
        try:
            print("\nStarting Aubio BPM Detector...")
            self.is_running = True
            
            device_id = self.find_vb_cable()
            device_info = sd.query_devices(device_id if device_id is not None else None, kind='input')
            channels = min(device_info['max_input_channels'], 2)
            
            print(f"Using device: {device_info['name']}")
            print(f"Channels: {channels}")
            
            with sd.InputStream(
                callback=self.audio_callback,
                channels=channels,
                samplerate=self.sample_rate,
                blocksize=self.block_size * 3,  # Using the multiplier
                device=device_id
            ):
                print("\nStarting BPM detection...")
                
                while self.is_running:
                    try:
                        audio_data = self.audio_queue.get(timeout=0.5)
                        results = self.analyze_tempo(audio_data)
                        
                        print(f"BPM: {results['bpm']:.1f}")
                        print(f"Confidence: {results['confidence']:.2f}")
                        print(f"Number of beats detected: {len(results['beat_times'])}")
                        print("---")
                        
                    except queue.Empty:
                        continue
                    except Exception as e:
                        print(f"Processing error: {e}")
                    
        except KeyboardInterrupt:
            print("\nStopping...")
        finally:
            self.cleanup()

if __name__ == "__main__":
    detector = AubioBPMDetector()
    detector.run()