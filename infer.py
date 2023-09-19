# source: https://github.com/coqui-ai/TTS/issues/2281 @padmalcom
# purpose of this file is to store the inference and model loading logic so we don't have to train every time.

from TTS.tts.configs.vits_config import VitsConfig
from TTS.config import load_config
from TTS.tts.models.vits import Vits
from TTS.utils.synthesizer import Synthesizer

import sys

def infer(m, c):
    config = load_config(c)
    model = Vits(config)

    config = load_config(c)
    model = Vits.init_from_config(config)
    model.load_checkpoint(config, m, eval=True)

    synth = Synthesizer(m, c)

    wav = synth.tts(
	    "Ask not what your country can do for you. Ask what you can do for your country.",
        reference_speaker_name="RN"
    )
    
    synth.save_wav(wav, "output_infer.wav")
    
# python3 infer.py /path/to/model /path/to/config

if __name__ == "__main__":
    if len(sys.argv) > 2:
        model = sys.argv[1]
        config = sys.argv[2]
        infer(model, config)
    else:
        exit

'''
from TTS.api import TTS

from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.models.vits import Vits
from TTS.config import load_config
from TTS.utils.synthesizer import Synthesizer

config = VitsConfig()
model = Vits(config)

config = load_config("config.json")
model = Vits.init_from_config(config)
model.load_checkpoint(config, 'best_model_226300.pth', eval=True)

speakers_file_path = None
language_ids_file_path = None
vocoder_path = None
vocoder_config_path = None
encoder_path = None
encoder_config_path = None
cuda = True

synthesizer = Synthesizer("best_model_226300.pth", "config.json", speakers_file_path, language_ids_file_path, vocoder_path, vocoder_config_path, encoder_path, encoder_config_path, cuda)

speaker_idx = None
language_idx = None
speaker_wav = None
reference_wav = None
style_wav = None
style_text = None
reference_speaker_name = None
wav = synthesizer.tts(
	"Das ist ein einfacher Test.",
	speaker_idx,
	language_idx,
	speaker_wav,
	reference_wav,
	style_wav,
	style_text,
	reference_speaker_name
)
synthesizer.save_wav(wav, "output.wav")
'''
