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
from bpm_detector import BPMDetector
from typing import Dict, List, Optional, Any

sys.stdout.reconfigure(line_buffering=True)

class FF3AudioBot:
    def __init__(self, target_battles: Optional[int] = None):
        # Audio parameters

        self.bpm_detector = BPMDetector()
        self.audio_stream = self.bpm_detector.setup_audio_stream()
        self.audio_stream.start()

        self.min_bpm_gate = 142  # Fixed baseline tempo for normal music
        self.max_bpm_gate = 160
        self.current_bpm = None
        #self.tempo_threshold = 25
        
        try:
            self.gamepad = vg.VX360Gamepad()
            print("Virtual Xbox controller created successfully")
        except Exception as e:
            print(f"Failed to create virtual controller: {e}")
            sys.exit(1)
        
        # Battle timings (in seconds)
        self.battle_first_wait = 12.0
        self.battle_second_wait = 10.0
        self.battle_final_wait = 23.0
        
        self.num_battles = 0
        self.target_battles = target_battles

        # Movement parameters
        self.button_cooldown = 0.1
        self.last_button_press = 0
        self.key_duration = 0.1
        self.direction = 'right'
        self.steps_taken = 0
        self.steps_per_direction = 3  # Number of steps before changing direction

        self.last_battle_end_time = None
        self.battle_cooldown = 2.0  # Seconds to wait after battle before new detection
        self.recalibration_samples = []  # Store tempos during recalibration
        self.recalibration_size = 5  # Number of samples to collect for recalibration

        # State control
        self.is_running = False
        self.in_battle = False
        self.start_time = None
        self.last_print_time = None
        self.calibration_time = 10

        self.quicksave_on = True

        self.after_battle_inc = 10
        
        # Register cleanup function
        atexit.register(self.cleanup)
        
    def cleanup(self):
        print("\nCleaning up...")
        self.is_running = False
        if hasattr(self, 'audio_stream'):
            self.audio_stream.stop()
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
    
    def process_audio(self):
        """Process audio data"""
        
        self.current_activity = 'audio_processing'

        if self.start_time is None:
            return False

        current_time = time.time()

        current_tempo = self.bpm_detector.get_bpm()

        if current_tempo is None:
            return False
            
        # Use median tempo for more stable detection
        # avg_tempo = np.median(self.tempo_buffer)

        if current_tempo:
            librosa_bpm = current_tempo['librosa']['bpm']
            aubio_bpm = current_tempo['aubio']['bpm']
            self.current_bpm = librosa_bpm
            if self.last_print_time is None or current_time - self.last_print_time >= 3:
                print(f"Current tempo Librosa: {current_tempo['librosa']['bpm']:.1f} BPM", flush=True)
                #print(f"Current tempo Aubio: {current_tempo['aubio']['bpm']:.1f} BPM", flush=True)
                print(f"--------------------------", flush=True)
                self.last_print_time = current_time

            if librosa_bpm > self.max_bpm_gate: #and aubio_bpm > self.min_bpm_gate:
                return True
                
            return False
        
            
        return False
    
    def get_bpm(self):
        """Get BPM readings directly from the BPM detector"""
        try:
            return self.bpm_detector.get_bpm()
        except Exception as e:
            print(f"Error getting BPM: {e}")
            return None
    
    def handle_battle(self):
        """Execute the battle macro sequence"""
        try:
            print("Executing battle sequence...")
            self.num_battles += 1
            
            # Initial battle commands
            time.sleep(0.1)  # Account for ambush
            self.press_button('X')  # Cancel/back
            # self.press_button('DPAD_RIGHT')
            # self.press_button('A')  # Confirm
            # self.press_button('DPAD_RIGHT')
            # self.press_button('A')
            # self.press_button('DPAD_DOWN')
            # self.press_button('A')
            # self.press_button('A')
            # self.press_button('A')
            # self.press_button('A')
            # self.press_button('DPAD_RIGHT')
            # self.press_button('A')
            # self.press_button('X')  # AutoBattle (FF:PixR)
            
            # # First wait period
            # print("Defending...", flush=True)
            # time.sleep(self.battle_first_wait)
            
            # self.press_button('X')
            
            # print("Wait to attack...", flush=True)
            # time.sleep(self.battle_second_wait)
            
            # # Final battle commands
            # for _ in range(8):
            #     self.press_button('A')
            # self.press_button('X')
            
            print("Attacking...", flush=True)
            time.sleep(self.battle_final_wait)
            self.press_button('A')
            self.press_button('A')
            print(f"{self.num_battles} battles complete!")
            
            self.after_battle_actions()
            # Quicksave sequence
            # time.sleep(3)
            # self.press_button('Y')  # Menu
            # for _ in range(3):
            #     self.press_button('DPAD_UP')
            #     time.sleep(0.2)
            # self.press_button('A')
            # time.sleep(0.2)
            # self.press_button('DPAD_LEFT')
            # time.sleep(0.2)
            # self.press_button('A')
            # time.sleep(0.2)
            # self.press_button('A')
            # time.sleep(0.2)
            # self.press_button('B')
            
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

    def after_battle_actions(self):
        """Custom actions for inbetween battles"""
        try:
            if self.quicksave_on:
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
            if self.num_battles % self.after_battle_inc == 0:
                # compelete custom macros
                print("Executing custom macro")
                time.sleep(0.1)
                self.press_button('Y')
                self.press_button('DPAD_DOWN')
                self.press_button('A')
                self.press_button('A')
                self.press_button('A')
                self.press_button('DPAD_DOWN')
                self.press_button('DPAD_DOWN')
                self.press_button('DPAD_RIGHT')
                self.press_button('A')
                self.press_button('DPAD_LEFT')
                self.press_button('A')
                for _ in range(5):
                    self.press_button('B')

        except Exception as e:
            print(f"Movement error: {e}")
              

    def audio_monitoring_thread(self):
        """Thread for continuous audio monitoring"""
        
        while self.is_running:
            try:
                self.check_exit_conditions()
                if self.last_battle_end_time is not None:
                    self.process_audio()
                    cooldown_elapsed = time.time() - self.last_battle_end_time
                    if cooldown_elapsed < self.battle_cooldown:
                        continue
                    else:
                        self.last_battle_end_time = None
                        print("Cooldown complete")
                
                if self.process_audio() and not self.in_battle:
                    print("Battle music detected!", flush=True)
                    self.in_battle = True
                    self.handle_battle()
                    self.in_battle = False
                    self.last_battle_end_time = time.time()  # Set cooldown time start
                    print("Returning to exploration...", flush=True)
                        
            except Exception as e:
                print(f"Audio processing error: {e}")
                self.cleanup()
                sys.exit(1)
    
    def run(self):
        """Main loop with enhanced pause handling"""
        
        self.is_running = True
        self.start_time = time.time()
        
        audio_thread = threading.Thread(target=self.audio_monitoring_thread)
        audio_thread.daemon = True
        audio_thread.start()
        time.sleep(10)
        
        try:
            while self.is_running:
                if self.num_battles == self.target_battles:
                    print(f"{self.num_battles} complete. Stopping bot")
                    self.cleanup()
                    sys.exit(0)

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
    pyautogui.FAILSAFE = False

    print("Starting in 5 seconds...")
    print("Controls:")
    print("- Press 'ESC' key to stop the bot")
    print("- Press Ctrl+C in this window to stop")
    print("\nHow many battles would you like to complete?")
    print("(Enter a number, or press Enter for unlimited)")
    time.sleep(5)
    
    try:
        target = input("> ").strip()
        target_battles = int(target) if target else None
        time.sleep(5)
        bot = FF3AudioBot(target_battles)
        bot.run()
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)