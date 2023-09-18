# Abhirav Joshi (@abhiravjoshi)
# Sumedh Garimella (@hdemusg)

# Purpose of main.py: generate dataset and train Vits model

import openai
import sys
import io
import os
from pprint import pprint

from trainer import Trainer, TrainerArgs

from TTS.tts.configs.shared_configs import BaseDatasetConfig, CharactersConfig
from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.vits import Vits, VitsArgs, VitsAudioConfig
from TTS.tts.utils.speakers import SpeakerManager
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio import AudioProcessor
from TTS.config.shared_configs import BaseAudioConfig

API_KEY = None

def main():
    
    # commented below so OpenAI doesn't charge us every time we run this script
    '''
    # Processing training data and creating txt file corpus
    with open("apikey.txt", 'r') as f:
        API_KEY = f.read()

    transcriptions = {}
    openai.api_key = API_KEY  # supply your API key however you choose
    files = ['training' + f for f in os.listdir('training/')]
    files.remove('training/.DS_Store')
    for fn in files: 
        f = open(fn, "rb")
        transcript = openai.Audio.transcribe("whisper-1", f)
        filename = fn.split('.')[0]
        transcriptions[filename] = transcript.text
        
    with open('transcript.txt', 'w') as f:
        for t in transcriptions:
            line = t[9:] + "|" + transcriptions[t]
            f.write(line + '\n')
    '''

    # Code Source: https://medium.com/@zahrizhalali/crafting-your-custom-text-to-speech-model-f151b0c9cba2
    # Credit to Zahrizhal Ali for pointing us in the direction of Coqui TTS. 

    dataset_config = BaseDatasetConfig(
        formatter="ljspeech", meta_file_train="../transcript.txt", path=os.path.join(os.getcwd(), "dataset/training/"))

    audio_config = BaseAudioConfig()

    tune_params={
        'num_mels': 80,          # In general, you don't need to change this. 
        'fft_size': 2400,        # In general, you don't need to change this.
        'frame_length_ms': 50, 
        'frame_shift_ms': 12.5,
        'sample_rate': 44100,    # This must match the sample rate of the dataset.
        'hop_length': 256,       # In general, you don't need to change this.
        'win_length': 1024,      # In general, you don't need to change this.
        'preemphasis': 0.97,     # In general, 0 gives better voice recovery but makes training harder. If your model does not train, try 0.97 - 0.99.
        'min_level_db': -100,
        'ref_level_db': 5,       # The base DB; increase until all background noise is removed in the spectrogram, then lower until you hear better speech below.
        'power': 1.5,            # Change this value and listen to the synthesized voice. 1.2 - 1.5 are resonable values.
        'griffin_lim_iters': 60, # Quality does not improve for values > 60
        'mel_fmin': 0.0,         # Adjust this and check mel-spectrogram-based voice synthesis below.
        'mel_fmax': 0.85,      # Adjust this and check mel-spectrogram-based voice synthesis below.
        'do_trim_silence': True  # If you dataset has some silience at the beginning or end, this trims it. Check the AP.load_wav() below,if it causes any difference for the loaded audio file.
    }

    # These options have to be forced off in order to avoid errors about the 
    # pre-calculated not matching the options being tuned.
    reset={
        'signal_norm': True,  # check this if you want to test normalization parameters.
        'stats_path': None,
        'symmetric_norm': False,
        'max_norm': 1,
        'clip_norm': True,
    }

    # Override select parts of loaded config with parameters above
    tuned_config = audio_config.copy()
    tuned_config.update(reset)
    tuned_config.update(tune_params)

    

    #audio_config = VitsAudioConfig(
    #    sample_rate=44100, win_length=1024, hop_length=256, num_mels=80, mel_fmin=0, mel_fmax=8000.00, preemphasis=0.95, rel_level_db=5
    #)

    character_config = CharactersConfig(
        characters_class= "TTS.tts.models.vits.VitsCharacters",
        characters= "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz1234567890",
        punctuations=" !,.?-'",
        pad= "<PAD>",
        eos= "<EOS>",
        bos= "<BOS>",
        blank= "<BLNK>",
    )

    config = VitsConfig(
        audio=audio_config,
        characters=character_config,
        run_name="vits_vctk",
        batch_size=16,
        eval_batch_size=4,
        eval_split_size=0.25,
        num_loader_workers=4,
        num_eval_loader_workers=4,
        run_eval=True,
        test_delay_epochs=0,
        epochs=10,
        text_cleaner="basic_cleaners",
        use_phonemes=False,
        phoneme_language="en-us",
        phoneme_cache_path=os.path.join(os.getcwd(), "phoneme_cache/"),
        compute_input_seq_cache=True,
        print_step=25,
        print_eval=False,
        save_best_after=10,
        save_checkpoints=True,
        save_all_best=True,
        mixed_precision=False,
        max_text_len=250,  # change this if you have a larger VRAM than 16GB
        output_path=os.getcwd(),
        datasets=[dataset_config],
        cudnn_benchmark=False,
        test_sentences=[
            ["These folks are good people... they deserve the same as the rest of us."],
            ["I'm a fighter, Pat. And I'll always be one."]
        ],
        
    )

    # Audio processor is used for feature extraction and audio I/O.
    # It mainly serves to the dataloader and the training loggers.
    ap = AudioProcessor(**tuned_config)

    # INITIALIZE THE TOKENIZER
    # Tokenizer is used to convert text to sequences of token IDs.
    # config is updated with the default characters if not defined in the config.
    tokenizer, config = TTSTokenizer.init_from_config(config)

    train_samples, eval_samples = load_tts_samples(dataset_config, 
                                                    eval_split=True, 
                                                    eval_split_size=config.eval_split_size,
                                                    formatter=formatter)
        
    print(len(train_samples), len(eval_samples))

    # init model
    model = Vits(config, ap, tokenizer, speaker_manager=None)

    # init the trainer and 🚀
    trainer = Trainer(
        TrainerArgs(),
        config,
        os.path.join(os.getcwd(), "output/"),
        model=model,
        train_samples=train_samples,
        eval_samples=eval_samples,
    )

    trainer.fit()

def formatter(root_path, manifest_file, **kwargs):  # pylint: disable=unused-argument
    """Assumes each line as ```<filename>|<transcription>```
    """
    txt_file = os.path.join(root_path, manifest_file)
    items = []
    speaker_name = "RN"
    with open(txt_file, "r", encoding="utf-8") as ttf:
        for line in ttf:
            cols = line.split("|")
            filename = "dataset/training/" + cols[0] + ".wav"
            wav_file = os.path.join(os.getcwd(), filename)
            text = cols[1].replace('\n', '')
            # print(text)
            items.append({"text":text, "audio_file":wav_file, "speaker_name":speaker_name, "root_path": root_path, "language": "eng"})
    return items


if __name__ == "__main__":
    main()