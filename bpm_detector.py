import numpy as np
import sounddevice as sd
import queue
import aubio
import librosa

class BPMDetector:
    def __init__(self):
        # Audio parameters
        self.sample_rate = 44100
        self.block_size = int(44100 * 3)  # 3 seconds of audio
        self.hop_size = 256  # Using the optimized hop size we found
        self.audio_queue = queue.Queue()
        self.volume_threshold = 0.01
        
        # Initialize aubio tempo detection with 'complex' method
        self.tempo = aubio.tempo(
            method="complex",
            buf_size=self.hop_size * 4,
            hop_size=self.hop_size,
            samplerate=self.sample_rate
        )

    def audio_callback(self, indata, frames, time, status):
        """Callback for sounddevice"""
        if status:
            print(status)
        self.audio_queue.put_nowait(indata.copy())

    def analyze_tempo_aubio(self, audio_data):
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
            'num_beats': len(beats)
        }

    def analyze_tempo_librosa(self, audio_data):
        """Calculate tempo using librosa with settings from original bot"""
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)
        
        # Convert to mono float32 array
        audio_mono = audio_data.astype(np.float32)
        
        # Calculate RMS volume
        rms_volume = np.sqrt(np.mean(audio_mono**2))
        
        # Only calculate tempo if volume is above threshold
        if rms_volume < self.volume_threshold:
            return {'bpm': 0.0}
            
        # Calculate onset envelope
        onset_env = librosa.onset.onset_strength(
            y=audio_mono, 
            sr=self.sample_rate,
            hop_length=512,
            aggregate=np.median,
            fmax=4000
        )
        
        # Get tempo using default settings
        tempo = librosa.beat.beat_track(
            onset_envelope=onset_env, 
            sr=self.sample_rate,
            hop_length=512
        )[0]
        
        return {'bpm': float(tempo)}

    def find_vb_cable(self):
        """Find VB-Cable audio input"""
        devices = sd.query_devices()
        for i, device in enumerate(devices):
            if (device['max_input_channels'] > 0 and 
                any(name in device['name'].lower() for name in ['vb', 'cable', 'virtual'])):
                print(f"Found VB-Cable: {device['name']}")
                return i
        print("VB-Cable not found, using default input")
        return None

    def setup_audio_stream(self):
        """Setup and return audio stream"""
        device_id = self.find_vb_cable()
        device_info = sd.query_devices(device_id if device_id is not None else None, kind='input')
        channels = min(device_info['max_input_channels'], 2)
        
        return sd.InputStream(
            callback=self.audio_callback,
            channels=channels,
            samplerate=self.sample_rate,
            blocksize=self.block_size * 3,
            device=device_id
        )

    def get_bpm(self):
        """Get BPM readings from both methods"""
        try:
            audio_data = self.audio_queue.get(timeout=0.5)
            aubio_results = self.analyze_tempo_aubio(audio_data)
            librosa_results = self.analyze_tempo_librosa(audio_data)
            
            return {
                'aubio': aubio_results,
                'librosa': librosa_results
            }
        except queue.Empty:
            return None
        except Exception as e:
            print(f"BPM detection error: {e}")
            return None

# For testing the module directly
if __name__ == "__main__":
    detector = BPMDetector()
    
    print("\nStarting dual BPM detection test...")
    print("Press Ctrl+C to stop")
    
    with detector.setup_audio_stream():
        try:
            while True:
                results = detector.get_bpm()
                if results:
                    print(f"Librosa BPM: {results['librosa']['bpm']:.1f}")
                    print(f"Aubio BPM: {results['aubio']['bpm']:.1f}")
                    print(f"Aubio Confidence: {results['aubio']['confidence']:.2f}")
                    print(f"Aubio Beats: {results['aubio']['num_beats']}")
                    print("---")
        except KeyboardInterrupt:
            print("\nStopping...")
    def __init__(self):
        # Audio parameters
        self.sample_rate = 44100
        self.block_size = int(44100 * 3)  # 3 seconds of audio
        self.hop_size = 512
        self.audio_queue = queue.Queue()
        
        # Initialize aubio tempo detection with 'complex' method
        self.tempo = aubio.tempo(
            method="complex",
            buf_size=self.hop_size * 4,
            hop_size=self.hop_size,
            samplerate=self.sample_rate
        )

    def audio_callback(self, indata, frames, time, status):
        """Callback for sounddevice"""
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
            'num_beats': len(beats)
        }

    def find_vb_cable(self):
        """Find VB-Cable audio input"""
        devices = sd.query_devices()
        for i, device in enumerate(devices):
            if (device['max_input_channels'] > 0 and 
                any(name in device['name'].lower() for name in ['vb', 'cable', 'virtual'])):
                print(f"Found VB-Cable: {device['name']}")
                return i
        print("VB-Cable not found, using default input")
        return None

    def setup_audio_stream(self):
        """Setup and return audio stream"""
        device_id = self.find_vb_cable()
        device_info = sd.query_devices(device_id if device_id is not None else None, kind='input')
        channels = min(device_info['max_input_channels'], 2)
        
        return sd.InputStream(
            callback=self.audio_callback,
            channels=channels,
            samplerate=self.sample_rate,
            blocksize=self.block_size * 3,
            device=device_id
        )

    def get_bpm(self):
        """Get single BPM reading from queue"""
        try:
            audio_data = self.audio_queue.get(timeout=0.5)
            return self.analyze_tempo(audio_data)
        except queue.Empty:
            return None
        except Exception as e:
            print(f"BPM detection error: {e}")
            return None

# For testing the module directly
if __name__ == "__main__":
    detector = BPMDetector()
    
    print("\nStarting BPM detection test...")
    print("Press Ctrl+C to stop")
    
    with detector.setup_audio_stream():
        try:
            while True:
                results = detector.get_bpm()
                if results:
                    print(f"BPM: {results['bpm']:.1f}")
                    print(f"Confidence: {results['confidence']:.2f}")
                    print(f"Beats detected: {results['num_beats']}")
                    print("---")
        except KeyboardInterrupt:
            print("\nStopping...")