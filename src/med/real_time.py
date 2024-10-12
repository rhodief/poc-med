import torch
import numpy as np
import pyaudio
import whisper
import wave

# Set up the audio stream
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
RECORD_SECONDS = 5  # Duration to record

torch.cuda.empty_cache()

# Load the model
print('Iniciando o Modelo')
model = whisper.load_model('medium')

input('Aperte enter para começar a gravar')

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

# Save the recorded audio as a WAV file
output_filename = './audios/temp.wav'

with wave.open(output_filename, 'wb') as wf:
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(audio_stream.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
del stream
del audio_stream

# passing to whisper
audio = whisper.load_audio(output_filename)
audio = whisper.pad_or_trim(audio)
# detect the spoken language

mel = whisper.log_mel_spectrogram(audio).to(model.device)
#_, probs = model.detect_language(mel)
# decode the audio
options = whisper.DecodingOptions()

torch.cuda.empty_cache()

result = whisper.decode(model, mel, options)
text = result.text
del audio
del mel
del result
del model
torch.cuda.empty_cache()
# print the recognized text
print("Transcription:", text)

