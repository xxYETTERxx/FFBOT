import numpy as np
import sounddevice as sd
import time
import queue
import threading
import sys
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
        self.baseline_tempo = 115
        self.tempo_threshold = 25
        self.tempo_buffer = []
        self.buffer_size = 5
        self.calibration_tempos = []
        self.tempo_offset = 5
        
        # Virtual controller setup
        try:
            self.gamepad = vg.VX360Gamepad()
            print("Virtual Xbox controller created successfully")
        except Exception as e:
            print(f"Failed to create virtual controller: {e}")
            sys.exit(1)
            
        # Movement and battle parameters
        self.button_cooldown = 0.1
        self.last_button_press = 0
        self.direction = 'right'
        self.steps_taken = 0
        self.steps_per_direction = 3
        self.battle_first_wait = 21.0
        self.battle_second_wait = 10.0
        self.battle_final_wait = 13.0
        
        # State control
        self.is_running = False
        self.in_battle = False
        
        # Register cleanup
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
    
    def handle_battle(self):
        """Execute battle sequence using virtual controller inputs"""
        try:
            print("Executing battle sequence...")
            
            # Initial battle commands
            time.sleep(5)  # Account for ambush
            self.press_button('B')  # Cancel/back
            self.press_button('DPAD_RIGHT')
            self.press_button('A')  # Confirm
            self.press_button('DPAD_RIGHT')
            self.press_button('A')
            self.press_button('DPAD_RIGHT')
            self.press_button('A')
            self.press_button('DPAD_RIGHT')
            self.press_button('A')
            self.press_button('X')  # Special command (q)
            
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
            time.sleep(0.2)
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

    def audio_callback(self, indata, frames, time, status):
        if status:
            print(status)
        self.audio_queue.put(indata.copy())
    
    def calculate_tempo(self, audio_data):
        """Calculate tempo from audio data using librosa"""
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)
        
        audio_mono = audio_data.astype(np.float32)
        rms_volume = np.sqrt(np.mean(audio_mono**2))
        
        if rms_volume < 0.01:  # Volume threshold
            return 0.0
            
        onset_env = librosa.onset.onset_strength(
            y=audio_mono, 
            sr=self.sample_rate,
            hop_length=512,
            aggregate=np.median,
            fmax=4000
        )
        
        tempo = librosa.beat.beat_track(
            onset_envelope=onset_env, 
            sr=self.sample_rate,
            hop_length=512
        )[0]
        
        return tempo
    
    def process_audio(self, audio_data):
        """Process audio data for battle detection"""
        if len(self.tempo_buffer) >= self.buffer_size:
            self.tempo_buffer.pop(0)
            
        current_tempo = self.calculate_tempo(audio_data)
        if current_tempo == 0.0:
            return False
            
        self.tempo_buffer.append(current_tempo)
        avg_tempo = np.median(self.tempo_buffer)
        
        # Debug printing every few seconds
        current_time = time.time()
        if not hasattr(self, 'last_print_time') or current_time - self.last_print_time >= 3:
            print(f"Current tempo: {avg_tempo:.1f} BPM", flush=True)
            print(f"Baseline tempo: {self.baseline_tempo:.1f} BPM", flush=True)
            print(f"Difference: {avg_tempo - self.baseline_tempo:.1f} BPM", flush=True)
            print("--------------------------", flush=True)
            self.last_print_time = current_time
        
        return avg_tempo > (self.baseline_tempo + self.tempo_threshold)

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
            # Get device info
            device_id = self.find_vb_cable()
            if device_id is not None:
                device_info = sd.query_devices(device_id)
            else:
                device_info = sd.query_devices(kind='input')

            # Use available channels (1 for mono, 2 for stereo)
            channels = min(device_info['max_input_channels'], 2)
            print(f"\nUsing audio device: {device_info['name']}")
            print(f"Channels: {channels}")
            print(f"Sample rate: {self.sample_rate}")
            
            with sd.InputStream(callback=self.audio_callback,
                              channels=channels,
                              samplerate=self.sample_rate,
                              blocksize=self.block_size *3,
                              device=device_id):
                while self.is_running:
                    try:
                        if keyboard.is_pressed('esc'):
                            print("\nEscape key pressed - stopping bot...")
                            self.is_running = False
                            break

                        audio_data = self.audio_queue.get(timeout=1)
                        if not self.in_battle and self.process_audio(audio_data):
                            print("Battle music detected!", flush=True)
                            self.in_battle = True
                            self.handle_battle()
                            
                    except queue.Empty:
                        continue
                        
        except Exception as e:
            print(f"Audio monitoring error: {e}")
            self.cleanup()
            sys.exit(1)

    def run(self):
        """Main bot loop"""
        print("\nStarting FF3 Bot...")
        print("Press 'ESC' to stop the bot")
        
        self.is_running = True
        
        # Start audio monitoring thread
        audio_thread = threading.Thread(target=self.audio_monitoring_thread)
        audio_thread.daemon = True
        audio_thread.start()
        
        try:
            while self.is_running:
                if keyboard.is_pressed('esc'):
                    print("\nEscape key pressed - stopping bot...")
                    break
                    
                if not self.in_battle:
                    self.move_character()
                    
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            print("\nBot stopped by user")
        finally:
            self.cleanup()
            sys.exit(0)

if __name__ == "__main__":
    print("Starting in 5 seconds...")
    print("Virtual controller will be created...")
    time.sleep(5)
    
    try:
        bot = FF3AudioBot()
        bot.run()
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)