import numpy as np
import sounddevice as sd
import pyautogui
import time
import queue
import threading
import sys
import keyboard
import atexit
import librosa
from scipy import signal

class FF3AudioBot:
    def __init__(self):
        # Audio parameters
        self.sample_rate = 44100
        self.block_size = 16384
        self.audio_queue = queue.Queue()
        self.baseline_tempo = 115  # Fixed baseline tempo for normal music
        self.tempo_threshold = 25
        self.tempo_buffer = []
        self.buffer_size = 5
        self.calibration_tempos = []
        self.tempo_offset = 5

        self.is_paused = False
        self.manual_pause = False
        self.last_user_input_time = None
        self.last_toggle_time = 0
        self.toggle_key = 'p'
        self.toggle_cooldown = 0.5
        self.pause_resume_delay = 15.0
        self.pause_flags = {
            'movement': False,
            'audio_processing': False,
            'battle_sequence': False,
            'key_monitoring': False
        }

        self.monitored_keys = [
            'left', 'right', 'up', 'down',  # Direction keys
            'enter', 'return',              # Confirm keys
            'backspace', 'tab',             # Menu navigation
            'q', 'w', 'e', 'r',            # Common game keys
            'space', 'p'                         # Additional controls
        ]

        self.current_activity = None

        self.consecutive_detections_needed = 3  # Need this many high tempos in a row
        self.current_consecutive_detections = 0
        self.volume_threshold = 0.01  # Minimum RMS volume to consider audio
        self.last_battle_end_time = None

        self.volume_buffer = []
        self.volume_buffer_size = 5
        
        # Wavelet analysis parameters
        self.levels = 4
        self.max_decimation = 2**(self.levels-1)
        self.min_bpm = 90  # Minimum expected BPM
        self.max_bpm = 180  # Maximum expected BPM
        
        # Battle timings (in seconds)
        self.battle_first_wait = 19.0
        self.battle_second_wait = 8.0
        self.battle_final_wait = 12.0

        # Movement parameters
        self.key_duration = 0.1
        self.direction = 'right'
        self.steps_taken = 0
        self.steps_per_direction = 3  # Number of steps before changing direction

        self.last_battle_end_time = None
        self.battle_cooldown = 5.0  # Seconds to wait after battle before new detection
        self.recalibration_samples = []  # Store tempos during recalibration
        self.recalibration_size = 3  # Number of samples to collect for recalibration

        self.toggle_key = 'p'
        self.manual_pause = False
        self.last_toggle_time = 0  # Prevent double-triggers
        self.toggle_cooldown = 0.5  # Seconds between toggle presses
            
        # State control
        self.is_running = False
        self.in_battle = False
        self.start_time = None
        self.last_print_time = None
        self.calibration_time = 10
        
        # Register cleanup function
        atexit.register(self.cleanup)
        
    def cleanup(self):
        """Ensure everything is properly cleaned up"""
        print("\nCleaning up...")
        self.is_running = False
        # Release all potentially held keys
        pyautogui.keyUp('left')
        pyautogui.keyUp('right')
        pyautogui.keyUp('enter')
        
    def check_exit_conditions(self):
        """Check if any exit conditions are met"""
        if keyboard.is_pressed('esc'):
            print("\nEscape key pressed - stopping bot...")
            self.cleanup()
            sys.exit(0)
            
    def audio_callback(self, indata, frames, time, status):
        """Callback function for audio stream"""
        if status:
            print(status)
        self.audio_queue.put(indata.copy())
    
    def calculate_tempo(self, audio_data):
        """Calculate tempo from audio data using librosa"""
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)
        
        # Convert to mono float32 array
        audio_mono = audio_data.astype(np.float32)
        
        # Calculate RMS volume
        rms_volume = np.sqrt(np.mean(audio_mono**2))
        
        # Store volume in buffer
        self.volume_buffer.append(rms_volume)
        if len(self.volume_buffer) > self.volume_buffer_size:
            self.volume_buffer.pop(0)
        
        # Only calculate tempo if volume is above threshold
        if rms_volume < self.volume_threshold:
            print("Volume too low, skipping tempo calculation")
            return 0.0
            
        # Calculate onset envelope
        onset_env = librosa.onset.onset_strength(
            y=audio_mono, 
            sr=self.sample_rate,
            hop_length=512,
            aggregate=np.median,
            fmax=8000
        )
        
        # Get tempo using default settings
        tempo = librosa.beat.tempo(
            onset_envelope=onset_env, 
            sr=self.sample_rate,
            hop_length=512,
            aggregate=None
        )[0]
        
        return tempo
    
    def process_audio(self, audio_data):
        """Process audio data with improved battle detection"""
        if not self.can_perform_activity('audio_processing'):
            return False
            
        self.current_activity = 'audio_processing'

        if self.start_time is None:
            return False

        current_time = time.time()
        
        # Check cooldown first
        if self.last_battle_end_time is not None:
            cooldown_elapsed = current_time - self.last_battle_end_time
            if cooldown_elapsed < self.battle_cooldown:
                print(f"In cooldown period ({cooldown_elapsed:.1f}s / {self.battle_cooldown:.1f}s)")
                self.current_consecutive_detections = 0  # Reset detection counter
                return False
            else:
                self.last_battle_end_time = None
                print("Cooldown complete")

        # Calculate current tempo
        current_tempo = self.calculate_tempo(audio_data)
        if current_tempo == 0.0:  # Volume too low
            self.current_consecutive_detections = 0
            return False
            
        # Add tempo to buffer
        self.tempo_buffer.append(current_tempo)
        if len(self.tempo_buffer) > self.buffer_size:
            self.tempo_buffer.pop(0)
        
        # Use median tempo for more stable detection
        avg_tempo = np.median(self.tempo_buffer)
        
        # Check if current tempo exceeds threshold
        tempo_difference = avg_tempo - self.baseline_tempo
        
        # Debug printing
        if self.last_print_time is None or current_time - self.last_print_time >= 3:
            print(f"Current tempo: {avg_tempo:.1f} BPM", flush=True)
            print(f"Baseline tempo: {self.baseline_tempo:.1f} BPM", flush=True)
            print(f"Tempo difference: {tempo_difference:.1f} BPM", flush=True)
            print(f"Consecutive detections: {self.current_consecutive_detections}", flush=True)
            print(f"Average volume: {np.mean(self.volume_buffer):.6f}", flush=True)
            self.last_print_time = current_time
        
        # Update detection counter
        if tempo_difference > self.tempo_threshold:
            self.current_consecutive_detections += 1
        else:
            self.current_consecutive_detections = 0
            
        
        # Only trigger battle if we have enough consecutive detections
        if self.current_consecutive_detections >= self.consecutive_detections_needed:
            self.current_consecutive_detections = 0  # Reset counter
            return True
            
        return False

    # Rest of the class remains the same (move_character, audio_monitoring_thread, run)
    def handle_battle(self):
        """Execute the battle macro sequence"""
        try:
            print("Executing battle sequence...")
            
            # Initial battle commands
            time.sleep(5) # account for ambush
            pyautogui.press('backspace')
            pyautogui.press('down')
            pyautogui.press('enter')
            pyautogui.press('right')
            pyautogui.press('enter')
            pyautogui.press('right')
            pyautogui.press('enter')
            pyautogui.press('right')
            pyautogui.press('enter')
            pyautogui.press('q')
            
            # First wait period
            print("Defending...", flush=True)
            time.sleep(self.battle_first_wait)
            
            # Mid-battle commands
            pyautogui.press('q')
            
            # Second wait period
            print("Wait to attack ...", flush=True)
            time.sleep(self.battle_second_wait)
            
            # Final battle commands
            for _ in range(8):
                pyautogui.press('enter')
            pyautogui.press('q')
            
            # Final wait before returning to exploration
            print("Attacking...", flush=True)
            time.sleep(self.battle_final_wait)
            pyautogui.press('enter')
            pyautogui.press('enter')
            print("Battle sequence completed", flush=True)
            time.sleep(2)

            # Quicksave game
            pyautogui.press('tab')
            time.sleep(.2)
            pyautogui.press('up')
            time.sleep(.2)
            pyautogui.press('up')
            time.sleep(.2)
            pyautogui.press('up')
            time.sleep(.2)
            pyautogui.press('enter')
            time.sleep(.2)
            pyautogui.press('left')
            time.sleep(.2)
            pyautogui.press('enter')
            time.sleep(.2)
            pyautogui.press('enter')
            time.sleep(.2)
            pyautogui.press('backspace')
            time.sleep(.2)
            
            # Explicitly start exploration mode
            self.in_battle = False
            self.steps_taken = 0
            self.direction = 'right'
        
        except Exception as e:
            print(f"Battle macro error: {e}")
            self.cleanup()
            sys.exit(1)

    def move_character(self):
        """Move the character in a fixed pattern: three steps right, three steps left"""
        if not self.can_perform_activity('movement'):
            return

        try:
            self.check_exit_conditions()
            self.current_activity = 'movement'
            
            # Take a step in current direction
            key = 'right' if self.direction == 'right' else 'left'
            pyautogui.keyDown(key)
            time.sleep(self.key_duration)
            pyautogui.keyUp(key)
            
            # Increment step counter
            self.steps_taken += 1
            
            # Change direction after three steps
            if self.steps_taken >= self.steps_per_direction:
                self.direction = 'left' if self.direction == 'right' else 'right'
                self.steps_taken = 0
            
        except Exception as e:
            print(f"Movement error: {e}")
            self.cleanup()
            sys.exit(1)
    
    def find_vb_cable(self):
        """Automatically find VB-Cable input device"""
        devices = sd.query_devices()
        for i, device in enumerate(devices):
            if device['max_input_channels'] > 0 and any(name.lower() in device['name'].lower() 
                for name in ['vb', 'cable', 'virtual']):
                print(f"Found VB-Cable: {device['name']}")
                return i
        print("VB-Cable not found, using default input")
        return None

    def audio_monitoring_thread(self):
        """Thread for continuous audio monitoring"""
        try:
            # Automatically select VB-Cable
            device_id = self.find_vb_cable()
            if device_id is not None:
                device_info = sd.query_devices(device_id)
            else:
                device_info = sd.query_devices(kind='input')
            
            channels = min(device_info['max_input_channels'], 2)
            print(f"\nUsing device: {device_info['name']}")
            print(f"Channels: {channels}")
            
            self.block_size = int(self.sample_rate * 2.0)
            print(f"Block size: {self.block_size} samples ({self.block_size/self.sample_rate:.3f} seconds)")
            
            with sd.InputStream(callback=self.audio_callback,
                            channels=channels,
                            samplerate=self.sample_rate,
                            blocksize=self.block_size,
                            device=device_id):
                while self.is_running:
                    try:
                        self.check_exit_conditions()
                        if self.is_paused:
                            time.sleep(0.1)
                            continue

                        # Get audio data from queue
                        try:
                            audio_data = self.audio_queue.get(timeout=1)
                        except queue.Empty:
                            continue

                        # Only process if not in battle and not in cooldown
                        if not self.in_battle:
                            if self.process_audio(audio_data):
                                print("Battle music detected!", flush=True)
                                self.in_battle = True
                                self.handle_battle()
                                self.in_battle = False
                                # Set cooldown time
                                self.last_battle_end_time = time.time()
                                # Only clear tempo buffer, don't reset baseline
                                self.tempo_buffer.clear()
                                print("Returning to exploration...", flush=True)
                                
                    except Exception as e:
                        print(f"Audio processing error: {e}")
                        self.cleanup()
                        sys.exit(1)
                        
        except Exception as e:
            print(f"Audio monitoring error: {e}")
            self.cleanup()
            sys.exit(1)

    def set_pause_state(self, paused):
        """Centralized method to handle pause state changes"""
        current_time = time.time()
        
        if paused:
            self.is_paused = True
            self.pause_flags = {k: True for k in self.pause_flags}
            
            # Release any held keys
            for key in ['left', 'right', 'up', 'down', 'enter', 'space']:
                pyautogui.keyUp(key)
                
            # Clear any pending activities
            self.current_activity = None
            
            if self.manual_pause:
                print("\nBot manually paused - Press 'p' to resume")
            else:
                print("\nBot auto-paused due to user input")
                self.last_user_input_time = current_time
                
        else:
            if self.manual_pause:  # Only unpause if it was manually paused
                self.is_paused = False
                self.manual_pause = False
                self.pause_flags = {k: False for k in self.pause_flags}
                self.last_user_input_time = None
                print("\nBot manually resumed")
            elif current_time - self.last_user_input_time >= self.pause_resume_delay:
                self.is_paused = False
                self.pause_flags = {k: False for k in self.pause_flags}
                self.last_user_input_time = None
                print(f"\nNo user input for {self.pause_resume_delay} seconds - Resuming bot")

    def check_user_input(self):
        """Enhanced user input checking"""
        current_time = time.time()
        
        # Check for toggle key (p key)
        if keyboard.is_pressed(self.toggle_key):
            if current_time - self.last_toggle_time > self.toggle_cooldown:
                self.manual_pause = not self.manual_pause
                self.last_toggle_time = current_time
                self.set_pause_state(self.manual_pause)
                time.sleep(0.1)
            return self.is_paused

        # If already manually paused, stay paused
        if self.manual_pause:
            return True

        # Check for game input
        if not self.is_paused:  # Only check if not already paused
            for key in self.monitored_keys:
                if keyboard.is_pressed(key):
                    self.set_pause_state(True)
                    return True

        # Check for auto-resume
        if self.is_paused and not self.manual_pause and self.last_user_input_time:
            self.set_pause_state(False)  # This handles the timing check

        return self.is_paused

    def can_perform_activity(self, activity):
        """Check if an activity can be performed"""
        return not self.is_paused and not self.pause_flags.get(activity, False)
    
    def run(self):
        """Main loop with enhanced pause handling"""
        print("\nStarting FF3 Bot with enhanced pause system...")
        print("Controls:")
        print("- Press 'p' to manually pause/resume")
        print("- Any game input will auto-pause")
        print(f"- Bot will auto-resume after {self.pause_resume_delay} seconds of no input")
        print("- Press 'ESC' to stop the bot")
        print("- Use Ctrl+C in this window to stop")
        
        self.is_running = True
        self.start_time = time.time()
        
        audio_thread = threading.Thread(target=self.audio_monitoring_thread)
        audio_thread.daemon = True
        audio_thread.start()
        
        try:
            while self.is_running:
                # Centralized pause check
                if self.check_user_input():
                    time.sleep(0.1)
                    continue
                    
                # Only proceed if not paused
                if not self.is_paused:
                    if not self.in_battle:
                        self.move_character()
                    time.sleep(0.1)
                    
        except KeyboardInterrupt:
            print("\nBot stopped by user")
        finally:
            self.cleanup()
            sys.exit(0)

if __name__ == "__main__":
    pyautogui.FAILSAFE = False
    
    print("WARNING: Make sure you can quickly switch to this window!")
    print("Starting in 5 seconds...")
    print("Controls:")
    print("- Press 'ESC' key to stop the bot")
    print("- Press Ctrl+C in this window to stop")
    time.sleep(5)
    
    try:
        bot = FF3AudioBot()
        bot.run()
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)