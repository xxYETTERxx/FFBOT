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
import vgamepad as vg
import keyboard

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
        
        try:
            self.gamepad = vg.VX360Gamepad()
            print("Virtual Xbox controller created successfully")
        except Exception as e:
            print(f"Failed to create virtual controller: {e}")
            sys.exit(1)

        # self.current_activity = None

        self.consecutive_detections_needed = 3  # Need this many high tempos in a row
        self.current_consecutive_detections = 0
        self.volume_threshold = 0.01  # Minimum RMS volume to consider audio
        self.last_battle_end_time = None

        self.volume_buffer = []
        self.volume_buffer_size = 5
        
        # Battle timings (in seconds)
        self.battle_first_wait = 21.0
        self.battle_second_wait = 10.0
        self.battle_final_wait = 14.0

        # Movement parameters
        self.button_cooldown = 0.1
        self.last_button_press = 0
        self.key_duration = 0.1
        self.direction = 'right'
        self.steps_taken = 0
        self.steps_per_direction = 3  # Number of steps before changing direction

        self.last_battle_end_time = None
        self.battle_cooldown = 5.0  # Seconds to wait after battle before new detection
        self.recalibration_samples = []  # Store tempos during recalibration
        self.recalibration_size = 5  # Number of samples to collect for recalibration

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
        if hasattr(self, 'gamepad'):
            self.gamepad.reset()
            time.sleep(0.1)
        
    def press_button(self, button, duration=0.05):
        """Press and release a virtual controller button"""
        current_time = time.time()
        if current_time - self.last_button_press < self.button_cooldown:
            time.sleep(self.button_cooldown) 

        try:
            # Map button names to actions
            if button == 'A':
                self.gamepad.press_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_A)
            elif button == 'B':
                self.gamepad.press_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_B)
            elif button == 'X':
                self.gamepad.press_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_X)
            elif button == 'Y':
                self.gamepad.press_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_Y)
            elif button == 'START':
                self.gamepad.press_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_START)
            elif button == 'BACK':
                self.gamepad.press_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_BACK)
            elif button == 'DPAD_UP':
                self.gamepad.press_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_UP)
            elif button == 'DPAD_DOWN':
                self.gamepad.press_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_DOWN)
            elif button == 'DPAD_LEFT':
                self.gamepad.press_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_LEFT)
            elif button == 'DPAD_RIGHT':
                self.gamepad.press_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_RIGHT)

            self.gamepad.update()
            time.sleep(duration)
            
            # Release the button
            if button == 'A':
                self.gamepad.release_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_A)
            elif button == 'B':
                self.gamepad.release_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_B)
            elif button == 'X':
                self.gamepad.release_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_X)
            elif button == 'Y':
                self.gamepad.release_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_Y)
            elif button == 'START':
                self.gamepad.release_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_START)
            elif button == 'BACK':
                self.gamepad.release_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_BACK)
            elif button == 'DPAD_UP':
                self.gamepad.release_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_UP)
            elif button == 'DPAD_DOWN':
                self.gamepad.release_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_DOWN)
            elif button == 'DPAD_LEFT':
                self.gamepad.release_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_LEFT)
            elif button == 'DPAD_RIGHT':
                self.gamepad.release_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_RIGHT)
                
            self.gamepad.update()
            self.last_button_press = time.time()
            
        except Exception as e:
            print(f"Button press error: {e}")   

    
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
            fmax=4000
        )
        
        # Get tempo using default settings
        tempo = librosa.beat.beat_track(
            onset_envelope=onset_env, 
            sr=self.sample_rate,
            hop_length=512
        )[0]
        
        return tempo
    
    def process_audio(self, audio_data):
        """Process audio data with improved battle detection"""
        
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
        current_tempo = self.calculate_tempo(audio_data)  # <- pretend this is a module call instead
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
            print(f"--------------------------", flush=True)
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
    
    def handle_battle(self):
        """Execute the battle macro sequence"""
        try:
            print("Executing battle sequence...")
            
            # Initial battle commands
            time.sleep(5)  # Account for ambush
            self.press_button('B')  # Cancel/back
            self.press_button('DPAD_RIGHT')
            self.press_button('A')  # Confirm
            self.press_button('DPAD_RIGHT')
            self.press_button('A')
            self.press_button('DPAD_DOWN')
            self.press_button('A')
            self.press_button('A')
            self.press_button('A')
            self.press_button('A')
            self.press_button('DPAD_RIGHT')
            self.press_button('A')
            self.press_button('X')  # AutoBattle (FF:PixR)
            
            # First wait period
            print("Defending...", flush=True)
            time.sleep(self.battle_first_wait)
            
            self.press_button('X')
            
            print("Wait to attack...", flush=True)
            time.sleep(self.battle_second_wait)
            
            # Final battle commands
            for _ in range(8):
                self.press_button('A')
            self.press_button('X')
            
            print("Attacking...", flush=True)
            time.sleep(self.battle_final_wait)
            self.press_button('A')
            self.press_button('A')
            
            # Quicksave sequence
            time.sleep(3)
            self.press_button('Y')  # Menu
            for _ in range(3):
                self.press_button('DPAD_UP')
                time.sleep(0.2)
            self.press_button('A')
            time.sleep(0.2)
            self.press_button('DPAD_LEFT')
            time.sleep(0.2)
            self.press_button('A')
            time.sleep(0.2)
            self.press_button('A')
            time.sleep(0.2)
            self.press_button('B')
            
            # Reset battle state
            self.in_battle = False
            self.steps_taken = 0
            self.direction = 'right'
            
        except Exception as e:
            print(f"Battle sequence error: {e}")

    def move_character(self):
        """Move the character using virtual d-pad"""
        try:
            button = 'DPAD_RIGHT' if self.direction == 'right' else 'DPAD_LEFT'
            self.press_button(button, duration=self.button_cooldown)
            
            self.steps_taken += 1
            if self.steps_taken >= self.steps_per_direction:
                self.direction = 'left' if self.direction == 'right' else 'right'
                self.steps_taken = 0
                
        except Exception as e:
            print(f"Movement error: {e}")
    
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
            
            self.block_size = int(self.sample_rate *2.5)
            print(f"Block size: {self.block_size} samples ({self.block_size/self.sample_rate:.3f} seconds)")
            
            with sd.InputStream(callback=self.audio_callback,
                            channels=channels,
                            samplerate=self.sample_rate,
                            blocksize=self.block_size,
                            device=device_id):
                while self.is_running:
                    try:
                        self.check_exit_conditions()
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
    
    def run(self):
        """Main loop with enhanced pause handling"""
        print("\nStarting FF3 Bot with enhanced pause system...")
        print("Controls:")
        print("- Press 'p' to manually pause/resume")
        print("- Any game input will auto-pause")
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
                if keyboard.is_pressed('esc'):
                    print("\nEscape key pressed - stopping bot...")
                    break
                    
                # Only proceed if not paused
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

#  try:
#     bpm_results = bpm_detector.get_bpm()  # Gets latest data from queue
#     if bpm_results:  # If we got a valid reading
#         if bpm_results['bpm'] > threshold:
#             # Handle battle detection
#             pass
# except queue.Empty:
#     # Queue was empty, just continue
#     pass