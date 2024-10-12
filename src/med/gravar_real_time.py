import pyaudio
import os
import wave
import numpy as np


# Constants
CHUNK = 1024  # Number of audio samples per chunk
FORMAT = pyaudio.paInt16  # Audio format
CHANNELS = 1  # Mono audio
RATE = 16000  # Sampling rate (samples per second)
THRESHOLD = 350  # Silence threshold (in audio amplitude)
SILENCE_LIMIT = 2  # Silence limit (in seconds)
AUDIO_FOLDER = './temp'  # Folder to save the audio files

# Create the audio folder if it does not exist
if not os.path.exists(AUDIO_FOLDER):
    os.makedirs(AUDIO_FOLDER)

print('Limpando a pasta')
all_files = os.listdir(AUDIO_FOLDER)
    # Filter out only the .wav files
wav_files = [file for file in all_files if file.endswith('.wav')]
for wav in wav_files:
    os.remove(os.path.join(AUDIO_FOLDER, wav))
input('Aperte qualquer botão para iniciar a gravação')
print('Iniciando Streaming')
# Create a PyAudio object
audio = pyaudio.PyAudio()

# Open a stream from the default microphone
stream = audio.open(format=FORMAT, channels=CHANNELS,
                    rate=RATE, input=True,
                    frames_per_buffer=CHUNK)

# Variables to store the audio data and the start time of a chunk
audio_data = []
start_time = None
chunk_count = 1
# Loop through the audio stream
while True:
    # Read a chunk of audio data from the stream
    chunk = stream.read(CHUNK)
    
    # Convert the raw audio data to an integer array
    data = np.frombuffer(chunk, dtype=np.int16)

    # Check if the audio amplitude is above the silence threshold
    amplitude = max(data)
    if amplitude > THRESHOLD:
        # If the speaker is speaking, add the audio data to the buffer
        audio_data.extend(data)
        # If this is the start of a new chunk, store the start time
        if start_time is None:
            start_time = pyaudio.get_sample_size(FORMAT) * len(audio_data) / RATE
    else:
        # If the speaker is silent, check if the silence limit has been reached
        if start_time is not None:
            # Calculate the duration of the chunk
            duration = pyaudio.get_sample_size(FORMAT) * len(audio_data) / RATE - start_time
            # If the duration is longer than the silence limit, save the chunk as a WAV file
            if duration >= SILENCE_LIMIT:
                # Create a new file name for the chunk based on the current time
                file_name = os.path.join(AUDIO_FOLDER, f'chunk_{chunk_count}.wav')
                chunk_count+=1
                # Open a new WAV file for writing
                with wave.open(file_name, 'wb') as wf:
                    # Set the WAV file parameters
                    wf.setnchannels(CHANNELS)
                    wf.setsampwidth(audio.get_sample_size(FORMAT))
                    wf.setframerate(RATE)
                    # Write the audio data to the WAV file
                    wf.writeframes(b''.join(audio_data))
                # Reset the audio buffer and start time
                audio_data = []
                start_time = None

# Clean up the PyAudio and audio stream objects
stream.stop_stream()
stream.close()
audio.terminate()