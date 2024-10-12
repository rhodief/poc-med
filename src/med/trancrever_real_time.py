import torch
import whisper
import os

torch.cuda.empty_cache()

# Load the model
output_filename = './temp'
print('Escutando pasta', output_filename)
print('Iniciando o Modelo')
model = whisper.load_model('medium')
options = whisper.DecodingOptions()
# passing to whisper
transcripted_audios = []
print('Ouvindo...')
while True:
    # List all files in the folder
    all_files = os.listdir(output_filename)
    # Filter out only the .wav files
    wav_files = [file for file in all_files if file.endswith('.wav') and file not in transcripted_audios]
    if len(wav_files) == 0: continue
    sorted(wav_files)
    audio_name = wav_files[0]
    audio = whisper.load_audio(os.path.join(output_filename, audio_name))
    audio = whisper.pad_or_trim(audio)
    # detect the spoken language
    mel = whisper.log_mel_spectrogram(audio).to(model.device)
    #_, probs = model.detect_language(mel)
    # decode the audio
    torch.cuda.empty_cache()

    result = whisper.decode(model, mel, options)
    text = result.text
    transcripted_audios.append(audio_name)
    del audio
    del mel
    del result
    torch.cuda.empty_cache()
    # print the recognized text
    print("Transcription:", text)
    if text.lower() == 'encerrar.':
        break

