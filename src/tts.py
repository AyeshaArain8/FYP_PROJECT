"""
Text-to-speech wrapper using pyttsx3 (fully offline, no API key needed).
"""

import pyttsx3


def speak(text: str):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()
    engine.stop()


def save_audio(text: str, out_path: str):
    engine = pyttsx3.init()
    engine.save_to_file(text, out_path)
    engine.runAndWait()
    engine.stop()


if __name__ == "__main__":
    import sys
    speak(" ".join(sys.argv[1:]))
