import torch
from aksharamukha import transliterate
import time
import os

# Loading model
model_tuple = torch.hub.load(repo_or_dir='snakers4/silero-models',
                             model='silero_tts',
                             language='indic',
                             speaker='v3_indic')

# Unpack the tuple (assuming the first element is the model)
model = model_tuple[0]  # Adjust if needed based on tuple contents

sample_rate = 24000
orig_text = "सुनो जी, अगर आपके बाल इसी रफ्तार से झड़ते रहे तो मैं तुम्हें तलाक दे दूँगी !!"
roman_text = transliterate.process('Devanagari', 'ISO', orig_text)  # Changed 'Hindi' to 'Devanagari'
print(roman_text)

# Generate timestamped filename
timestamp = time.strftime("%Y%m%d_%H%M%S")  # e.g., 20250531_232255
filename = f"hindi_tts_{timestamp}.wav"

# Try passing the filename to save_wav
try:
    audio_paths = model.save_wav(roman_text, speaker='hindi_male', sample_rate=sample_rate, audio_path=filename)
    print(f"Audio saved as: {audio_paths}")
except TypeError as e:
    print(f"TypeError: {e} - Falling back to default filename")
    try:
        audio_paths = model.save_wav(roman_text, speaker='hindi_male', sample_rate=sample_rate)
        print(f"Default audio saved as: {audio_paths}")
        if os.path.exists("test.wav"):
            os.rename("test.wav", filename)
            print(f"Audio renamed to: {filename}")
        else:
            print("Error: test.wav was not created.")
    except Exception as e:
        print(f"Error in save_wav or rename: {e}")