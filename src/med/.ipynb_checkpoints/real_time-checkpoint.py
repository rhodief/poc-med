import torch
import numpy as np
import pyaudio
import whisper

# Load the model
model = whisper.load_model('medium')

# Set up the audio stream
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
RECORD_SECONDS = 5  # Duration to record

audio_stream = pyaudio.PyAudio()

stream = audio_stream.open(format=FORMAT,
                           channels=CHANNELS,
                           rate=RATE,
                           input=True,
                           frames_per_buffer=CHUNK)

print("Start recording...")
# Record the audio
frames = []
for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    data = stream.read(CHUNK)
    np_data = np.frombuffer(data, dtype=np.int16)
    frames.append(np_data)

print("Finished recording")

# Stop the audio stream
stream.stop_stream()
stream.close()
audio_stream.terminate()

# Process the audio data
audio_data = np.concatenate(frames)
audio_data = audio_data.astype(np.float32)

# Pad or trim the audio data if needed
audio_data = whisper.pad_or_trim(audio_data)

# Transcribe the audio
transcription = model.predict(audio_data)

print("Transcription:", transcription)

del stream
del audio_stream
del audio_data
del transcription
# Free up GPU memory
torch.cuda.empty_cache()
