from __future__ import absolute_import, division, print_function, unicode_literals

import glob
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import argparse
import json
import torch
from scipy.io.wavfile import write
from env import AttrDict
from meldataset import mel_spectrogram, MAX_WAV_VALUE, load_wav
from models import Generator
from stft import TorchSTFT
import soundfile as sf

from Utils.JDC.model import JDCNet

h = None
device = None


def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")
    return checkpoint_dict


def get_mel(x):
    return mel_spectrogram(x, h.n_fft, h.num_mels, h.sampling_rate, h.hop_size, h.win_size, h.fmin, h.fmax)

def get_mel_24k(x):
    return mel_spectrogram(x, 1024, h.num_mels, 24000, 240, 1024, h.fmin, 8000)

def scan_checkpoint(cp_dir, prefix):
    pattern = os.path.join(cp_dir, prefix + '*')
    cp_list = glob.glob(pattern)
    if len(cp_list) == 0:
        return ''
    return sorted(cp_list)[-1]

F0_model = JDCNet(num_class=1, seq_len=192)

output_dir = "gen_from_wav"
os.makedirs(output_dir, exist_ok=True)
cp_path = "exp/v1_16k_to_48k/"
if not os.path.exists(cp_path):
    raise ValueError(f"cp_path not exists: {cp_path}")


with open(cp_path + "/config.json") as f:
    data = f.read()

json_config = json.loads(data)
h = AttrDict(json_config)

# device = torch.device('cuda:{:d}'.format(0))
# global device
if torch.cuda.is_available():
    torch.cuda.manual_seed(h.seed)
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

generator = Generator(h, F0_model).to(device)
stft = TorchSTFT(filter_length=h.gen_istft_n_fft, hop_length=h.gen_istft_hop_size, win_length=h.gen_istft_n_fft).to(device)

cp_g = scan_checkpoint(cp_path, 'g_')
state_dict_g = load_checkpoint(cp_g, device)
generator.load_state_dict(state_dict_g['generator'])
generator.remove_weight_norm()
_ = generator.eval()

# pick a file to resynthesize
path = os.path.join("LJSpeech-BZNSYP-16k", "BZNSYP-000001.wav")

wav, sr = load_wav(path, sr=24000)
# wav = wav / MAX_WAV_VALUE
wav = torch.FloatTensor(wav).to(device)
x = get_mel_24k(wav.unsqueeze(0))

with torch.no_grad():
    spec, phase = generator(x)
    y_g_hat = stft.inverse(spec, phase)
    audio = y_g_hat.squeeze()
    audio = audio * MAX_WAV_VALUE
    audio = audio.cpu().numpy().astype('int16')

output_file = os.path.join(output_dir, os.path.basename(path))
sf.write(output_file,audio,h.sampling_rate,"PCM_16")

# import IPython.display as ipd

# print('Synthesized:')
# display(ipd.Audio(audio, rate=24000))

# print('Original:')
# display(ipd.Audio(wav.cpu(), rate=24000))