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
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from enum import Enum

class GameState(Enum):
    EXPLORING = "exploring"
    BATTLE = "battle"
    PAUSED = "paused"

@dataclass
class AudioConfig:
    sample_rate: int = 44100
    block_size: int = 16384
    sample_length: int = 3
    sample_size: int = 3
    volume_threshold: float = 0.01
    baseline_tempo: float = 115
    tempo_threshold: float = 30
    buffer_size: int = 5
    consecutive_detections_needed: int = 3

@dataclass
class BattleConfig:
    pre_battle_wait: float = 3.0
    first_wait: float = 19.0
    second_wait: float = 8.0
    final_wait: float = 12.0
    cooldown: float = 5.0
    max_duration: float = 120.0
    cooldown: float = 5.0 

class AudioProcessor:
    def __init__(self, config: AudioConfig):
        self.config = config
        self.audio_queue = queue.Queue()
        self.tempo_buffer: List[float] = []
        self.volume_buffer: List[float] = []
        self.current_consecutive_detections: int = 0
        
    def calculate_tempo(self, audio_data: np.ndarray) -> float:
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)
        
        audio_mono = audio_data.astype(np.float32)
        rms_volume = np.sqrt(np.mean(audio_mono**2))
        
        self.volume_buffer.append(rms_volume)
        if len(self.volume_buffer) > self.config.buffer_size:
            self.volume_buffer.pop(0)
        
        if rms_volume < self.config.volume_threshold:
            return 0.0
            
        onset_env = librosa.onset.onset_strength(
            y=audio_mono,
            sr=self.config.sample_rate,
            hop_length=512,         
            aggregate=np.median,        
            fmax=4000,                          
        )
        
        tempo = librosa.beat.tempo(
            onset_envelope=onset_env,
            sr=self.config.sample_rate,
            hop_length=512,
            aggregate=None,
            start_bpm = self.config.baseline_tempo                   
        )[0]
        
        return tempo

    def detect_battle(self, audio_data: np.ndarray) -> bool:
        current_tempo = self.calculate_tempo(audio_data)
        if current_tempo == 0.0:
            self.current_consecutive_detections = 0
            return False
            
        self.tempo_buffer.append(current_tempo)
        if len(self.tempo_buffer) > self.config.buffer_size:
            self.tempo_buffer.pop(0)

        weights = np.linspace(0.5, 1.0, len(self.tempo_buffer))
        avg_tempo = np.average(self.tempo_buffer, weights=weights)
        tempo_difference = avg_tempo - self.config.baseline_tempo
        
        avg_tempo = np.median(self.tempo_buffer)
        tempo_difference = avg_tempo - self.config.baseline_tempo

        print(f"Current BPM: {avg_tempo:.1f}", flush=True)
        print(f"Baseline BPM: {self.config.baseline_tempo}", flush=True)
        print(f"Difference: {tempo_difference:.1f}", flush=True)
        print(f"Consecutive detections: {self.current_consecutive_detections}", flush=True)
        print("-" * 30, flush=True) 
        
        if tempo_difference > self.config.tempo_threshold:
            self.current_consecutive_detections += 1
        else:
            self.current_consecutive_detections = 0
            
        return self.current_consecutive_detections >= self.config.consecutive_detections_needed

class InputHandler:
    def __init__(self):
        self.toggle_key = 'p'
        self.last_toggle_time = 0
        self.toggle_cooldown = 0.5
        self.pause_resume_delay = 15.0
        self.last_user_input_time = None
        self.manual_pause = False
        
        self.monitored_keys = [
            'left', 'right', 'up', 'down',
            'enter', 'return', 'backspace', 'tab',
            'q', 'w', 'e', 'r', 'space', 'p'
        ]
    
    def check_exit(self) -> bool:
        if keyboard.is_pressed('esc'):
            print("\nEscape key pressed - stopping bot...")
            return True
        return False
    
    def should_pause(self) -> bool:
        current_time = time.time()
        
        # Check for manual toggle
        if keyboard.is_pressed(self.toggle_key):
            if current_time - self.last_toggle_time > self.toggle_cooldown:
                self.manual_pause = not self.manual_pause
                self.last_toggle_time = current_time
                if self.manual_pause:
                    print("\nBot manually paused - Press 'p' to resume")
                else:
                    print("\nBot resumed")
                return self.manual_pause
                
        # If manually paused, stay paused
        if self.manual_pause:
            return True
            
        # Check for game input
        if any(keyboard.is_pressed(key) for key in self.monitored_keys):
            self.last_user_input_time = current_time
            return True
            
        # Check for auto-resume
        if self.last_user_input_time:
            if current_time - self.last_user_input_time >= self.pause_resume_delay:
                self.last_user_input_time = None
                print(f"\nNo user input for {self.pause_resume_delay} seconds - Resuming bot")
                return False
            return True
            
        return False

class BattleManager:
    def __init__(self, config: BattleConfig):
        self.config = config
        self.battle_count = 0
        self.target_battles: Optional[int] = None
        self.battle_times: List[float] = []
        self.start_time: Optional[float] = None
        self.last_battle_end_time: Optional[float] = None
        self.in_cooldown = False
    
    def execute_battle_sequence(self):
        """Execute the optimized battle sequence"""
        def press_key(key: str, wait: float = 0.1):
            pyautogui.press(key)
            time.sleep(wait)
        
        try:
            # Initial battle setup
            self.in_cooldown = True
            time.sleep(self.config.pre_battle_wait)
            
            # Battle menu navigation
            for key in ['backspace', 'right', 'enter', 'right', 'enter', 
                       'down', 'enter', 'enter', 'enter', 'right', 
                       'enter', 'q']:
                press_key(key, 0.2)
            
            # First defense phase
            time.sleep(self.config.first_wait)
            press_key('q')
            
            # Attack preparation
            time.sleep(self.config.second_wait)
            
            # Execute attacks
            for _ in range(8):
                press_key('enter')
            press_key('q')
            
            # Battle completion
            time.sleep(self.config.final_wait)
            press_key('enter')
            press_key('enter')
            time.sleep(1)
            
            # Quicksave
            for key in ['tab', 'up', 'up', 'up', 'enter', 'left', 'enter', 
                       'enter', 'backspace']:
                press_key(key, 0.2)
            
            self.complete_battle()

            time.sleep(self.config.cooldown)
            self.in_cooldown = False
            
        except Exception as e:
            print(f"Battle sequence error: {e}")
            # Emergency battle exit
            press_key('q')
            for _ in range(3):
                press_key('enter')
    
    def complete_battle(self) -> bool:
        """Update battle statistics and check if target reached"""
        self.battle_count += 1
        self.battle_times.append(time.time())
        self.last_battle_end_time = time.time()
        
        elapsed_time = time.time() - self.start_time
        avg_time = elapsed_time / self.battle_count if self.battle_count > 0 else 0
        
        self.consecutive_detections = 0
        
        
        print(f"\nBattle #{self.battle_count} completed!")
        if self.target_battles:
            print(f"Battles remaining: {self.target_battles - self.battle_count}")
        print(f"Average time per battle: {avg_time:.1f} seconds")
        
        if self.target_battles and self.battle_count >= self.target_battles:
            print(f"\nTarget of {self.target_battles} battles reached!")
            return True
        return False

class MovementManager:
    def __init__(self):
        self.direction = 'right'
        self.steps_taken = 0
        self.steps_per_direction = 3
        self.key_duration = 0.1
    
    def move(self):
        """Execute movement pattern"""
        try:
            # Take a step in current direction
            pyautogui.keyDown(self.direction)
            time.sleep(self.key_duration)
            pyautogui.keyUp(self.direction)
            
            # Increment step counter
            self.steps_taken += 1
            
            # Change direction after set number of steps
            if self.steps_taken >= self.steps_per_direction:
                self.direction = 'left' if self.direction == 'right' else 'right'
                self.steps_taken = 0
                
        except Exception as e:
            print(f"Movement error: {e}")

class FF3Bot:
    def __init__(self, target_battles: Optional[int] = None):
        self.audio_config = AudioConfig()
        self.battle_config = BattleConfig()
        
        self.audio_processor = AudioProcessor(self.audio_config)
        self.input_handler = InputHandler()
        self.battle_manager = BattleManager(self.battle_config)
        self.movement_manager = MovementManager()
        
        self.battle_manager.target_battles = target_battles
        self.state = GameState.EXPLORING
        self.state = GameState.PAUSED
        self.is_running = False
        self.audio_initialized = False
        
        atexit.register(self.cleanup)
    
    def start(self):
        """Initialize and start the bot"""
        print("\nStarting FF3 Bot...")
        print("Controls:")
        print("- Press 'p' to manually pause/resume")
        print("- Any game input will auto-pause")
        print("- Bot will auto-resume after 15 seconds of no input")
        print("- Press 'ESC' to stop the bot")
        
        self.is_running = True
        self.battle_manager.start_time = time.time()
        
        # Start audio thread first
        audio_thread = threading.Thread(target=self.audio_monitoring_thread)
        audio_thread.daemon = True
        audio_thread.start()
        
        print("\nCalibrating audio baseline (10 seconds)...")
        self.calibrate_audio()
        
        print("\nAudio calibration complete. Starting bot movement...")
        self.state = GameState.EXPLORING  # Only now change to EXPLORING
        
        try:
            self.main_loop()
        except KeyboardInterrupt:
            print("\nBot stopped by user")
        finally:
            self.cleanup()

    def calibrate_audio(self):
        """Wait for initial audio calibration"""
        calibration_time = 10  # 10 seconds calibration
        start_time = time.time()
        
        while time.time() - start_time < calibration_time:
            remaining = int(calibration_time - (time.time() - start_time))
            print(f"\rCalibrating... {remaining} seconds remaining", end="", flush=True)
            time.sleep(1)
        
        print("\nInitial BPM values:")
        print(f"Baseline: {self.audio_config.baseline_tempo}")
        self.audio_initialized = True
    
    def main_loop(self):
        """Main bot execution loop"""
        while self.is_running:
            try:
                if self.input_handler.check_exit():
                    break
                
                # Check for pause state
                if self.input_handler.should_pause():
                    if self.state != GameState.PAUSED:
                        self.state = GameState.PAUSED
                    time.sleep(0.1)
                    continue
                elif self.state == GameState.PAUSED:
                    self.state = GameState.EXPLORING

                if not self.audio_initialized:
                    time.sleep(0.1)
                    continue
                
                if self.state == GameState.EXPLORING:
                    self.movement_manager.move()
                elif self.state == GameState.BATTLE:
                    self.battle_manager.execute_battle_sequence()
                    self.state = GameState.EXPLORING
                
                time.sleep(0.1)
                
            except Exception as e:
                print(f"Main loop error: {e}")
                continue
    
    def audio_monitoring_thread(self):
        """Audio monitoring and battle detection thread"""
        try:
            device_id = self.find_audio_device()
            
            with sd.InputStream(callback=self.audio_callback,
                              channels=2,
                              samplerate=self.audio_config.sample_rate,
                              blocksize=self.audio_config.block_size * self.audio_config.sample_length,
                              device=device_id):
                while self.is_running:
                    try:
                        if self.state == GameState.PAUSED:
                            time.sleep(0.1)
                            continue
                            
                        audio_data = self.audio_processor.audio_queue.get(timeout=1)

                        if not self.audio_initialized:
                            continue

                        if (self.battle_manager.in_cooldown):
                            continue
                        
                        if (self.state == GameState.EXPLORING and 
                            self.audio_processor.detect_battle(audio_data)):
                            print("\nBattle detected!")
                            self.state = GameState.BATTLE
                            
                    except queue.Empty:
                        continue
                    except Exception as e:
                        print(f"Audio processing error: {e}")
                        continue
                        
        except Exception as e:
            print(f"Audio thread error: {e}")
            self.cleanup()
            sys.exit(1)
    
    def find_audio_device(self) -> Optional[int]:
        """Find the VB-Cable audio device"""
        devices = sd.query_devices()
        for i, device in enumerate(devices):
            if (device['max_input_channels'] > 0 and 
                any(name.lower() in device['name'].lower() 
                    for name in ['vb', 'cable', 'virtual'])):
                print(f"Found VB-Cable: {device['name']}")
                return i
        print("VB-Cable not found, using default input")
        return None
    
    def audio_callback(self, indata: np.ndarray, frames: int, 
                      time_info: Any, status: Any):
        """Callback for audio stream"""
        if status:
            print(status)
        self.audio_processor.audio_queue.put(indata.copy())
    
    def cleanup(self):
        """Clean up resources and reset game state"""
        print("\nCleaning up...")
        self.is_running = False
        pyautogui.keyUp('left')
        pyautogui.keyUp('right')
        pyautogui.keyUp('enter')

if __name__ == "__main__":
    pyautogui.FAILSAFE = False
    
    print("WARNING: Make sure you can quickly switch to this window!")
    print("Starting in 5 seconds...")
    print("Controls:")
    print("- Press 'ESC' key to stop the bot")
    print("- Press Ctrl+C in this window to stop")
    print("\nHow many battles would you like to complete?")
    print("(Enter a number, or press Enter for unlimited)")
    
    try:
        target = input("> ").strip()
        target_battles = int(target) if target else None
        time.sleep(5)
        
        bot = FF3Bot(target_battles)
        bot.start()
    except ValueError:
        print("Invalid input. Please enter a number or press Enter.")
        sys.exit(1)
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)