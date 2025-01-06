**FFBot - Audio-Based RPG Battle Automation**
FFBot is a Python-based automation tool that detects battles in RPG games by monitoring the game's audio output. When battle music is detected, it executes predefined battle commands automatically.

**Prerequisites**
Python 3.8 or higher
VB-Cable Virtual Audio Device (or similar virtual audio cable software)
Windows OS (untested on other platforms)

**Required Python Packages**
bashCopypip install numpy sounddevice pyautogui keyboard librosa scipy
Setup

Install VB-Cable:

Download from VB-Cable website
Install following their instructions
Set VB-Cable as your game's audio output
Set your speakers/headphones to listen to VB-Cable


**Clone this repository:**

bashCopygit clone https://github.com/yourusername/ffbot.git
cd ffbot

**Install required packages:**

bashCopypip install -r requirements.txt
Usage

**Start your game**
Make sure audio is routing through VB-Cable
Run the bot:

bashCopypython ffbot.py

**Controls**
Press 'ESC' to stop the bot
Any keyboard game input will auto-pause the bot
Bot will auto-resume after 15 seconds of no input

**Important Notes**

Please use responsibly and in accordance with game terms of service
Currently configured for FF3 but can be adapted for other RPGs
Make sure to position your character in a safe area before starting
Bot will automatically save after each battle

**Current Features**

Audio-based battle detection using BPM analysis
Automated battle sequence execution
Auto-save after battles
Basic exploration movement pattern
Pause/resume functionality
Emergency stop

**Planned Features**

Audio Frequency-based battle detection
Customizable macors for battle sequences
Customizable macros for outside of battle sequence with variable length of execution (use cure after 3 battles)
Configuration GUI
Support for multiple games/scenarios
improved detection accuracy

**Contributing**

Contributions are welcome! Please feel free to submit a Pull Request.
License
MIT License

**Disclaimer**

This tool is for educational purposes only. Use at your own risk. The creators are not responsible for any consequences of using this tool.
