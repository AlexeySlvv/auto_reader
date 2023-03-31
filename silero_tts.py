import os
import torch
from pydub import AudioSegment
import nltk

nltk.download('punkt')

device = torch.device('cpu')
torch.set_num_threads(4)
local_file = 'v3_1_ru.pt'

if not os.path.isfile(local_file):
    torch.hub.download_url_to_file(
        'https://models.silero.ai/models/tts/ru/v3_1_ru.pt', local_file)

model = torch.package.PackageImporter(
    local_file).load_pickle("tts_models", "model")
model.to(device)


INPUT_TXT = './txt/input.txt'
INPUT_DIR = os.path.dirname(os.path.realpath(INPUT_TXT))
BASE_NAME = os.path.splitext(os.path.basename(INPUT_TXT))[0]


with open(INPUT_TXT, mode='r+', encoding='utf-8') as txt_in:
    example_text = txt_in.read()


sample_rate = 48000
speakers = [
    'aidar',
    'baya',
    'eugene',
    'kseniya',
    'xenia',
    # 'random',
]

for speaker in speakers:
    audio_lst = []

    for line in (l.strip() for l in example_text.split('\n')):
        if not line:
            audio_lst.append(AudioSegment.silent(duration=300))
            continue
        
        print(line)

        for sent in (nltk.tokenize.sent_tokenize(line, language='russian')):
            audio_paths = model.save_wav(text=sent,
                                        speaker=speaker,
                                        sample_rate=sample_rate)
            audio_lst.append(AudioSegment.from_file("test.wav", format="wav"))

    combined = AudioSegment.empty()
    for a in audio_lst:
        combined += a

    file_handle = combined.export(os.path.join(INPUT_DIR, f"{BASE_NAME}_{speaker}.mp3"), format="mp3")
