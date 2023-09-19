import glob

output_path = "vits_1epoch"

ckpts = sorted([f for f in glob.glob(output_path+"/*.pth")])
configs = sorted([f for f in glob.glob(output_path+"/config.json")])
save_file = output_path + "test_audio.wav"

print("ckpts: ",ckpts[0])
print("configs: ",configs[0])
print("Saved file_name: ",save_file)

import subprocess

text = "Hello, nice to meet you."

command = f"tts --text \"{text}\" --model_path \"{ckpts[0]}\" --config_path \"{configs[0]}\" --out_path \"{save_file}\""
subprocess.run(command, shell=True)

import IPython
import librosa

# Load the audio file and get the audio data and sampling rate
audio_data, sampling_rate = librosa.load(save_file, sr=None)

# Display the audio using IPython.display.Audio
IPython.display.Audio(data=audio_data, rate=sampling_rate)