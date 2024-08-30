# ESPnet ASR models
This repository provides trained ESPnet ASR models in "Japanese" and some examples. 
These models are developed mainly for academic research. 

## Features
* Support Japanese models
* Support Noise-robust models
* Support batch and streaming models
* Support Kanji-Katakana-Hiragana and Katakana (Syllable-like) character models

## Requirements
Python and ESPnet are required.
- espnet
- torchaudio
- espnet_model_zoo

Python3.10 is suitable for installation of ESPnet (as of 2024/8). 

## How to use
### Set up ESPnet
```
python3 -m venv venv
. venv/bin/activate
python3 -m pip install espnet torchaudio
python3 -m pip install -U espnet_model_zoo
```

If you want to setup automatically, run the shell script "setup.sh".
This scripts creates virtual environment and install espnet.
```
sh setup.sh
```

### Case: Normal ASR (batch)
Use "batch.py" for batch processing.
```
. venv/bin/activate             # only once
python3 batch.py sample.wav
```

Loading model is achieved by calling "from_pretrained" method of "Speech2Text" class. 
```
import torchaudio
from espnet2.bin.asr_inference import Speech2Text

args = usage()   # get parameter settings. This needs to be defined. 

hfrepo = 'ouktlab/espnet_robustcsj_asr_train_asr_transformer_lm_rnn'

model = Speech2Text.from_pretrained(
    hfrepo,
    device=args.device,
    token_type='char',
    bpemodel=None,
    maxlenratio=0.0,
    minlenratio=0.0,
    beam_size=args.beam_size,
    nbest=args.nbest,
    ctc_weight=args.ctc_weight,
    lm_weight=args.lm_weight,
    penalty=0.0
)

s, fs = torchaudio.load(args.filename)
s = s.squeeze()

results = model(s)
print(f'Result: {results[0][0]}')
```

### Case: Streaming ASR (GPU processing is recommended)
Use "streaming.py" for "low-latency" streaming processing. The total processing cost of it may be larger than that of batch processing. This model may be suitable for spoken dialogue system.
```
. venv/bin/activate            # only once
python3 streaming.py sample.wav
```

Bacause "from_pretrained" method has not been implemented in "Speech2TextStreaming" class yet, its wrapper class is defined and used. The following is an example code of streaming recognition.
```
args = usage()

hfrepo = 'ouktlab/espnet_streaming_robustcsj_asr_train_asr_transformer_lm_rnn'

model = Speech2TextStreamingInterface.from_pretrained(
    hfrepo,
    device=args.device,
    token_type='char',
    bpemodel=None,
    maxlenratio=0.0,
    minlenratio=0.0,
    beam_size=args.beam_size,
    nbest=args.nbest,
    ctc_weight=args.ctc_weight,
    lm_weight=args.lm_weight,
    penalty=0.0
)

s, fs = torchaudio.load(args.filename)
s = s.squeeze()

segment_len = 1600
for pos in range(0, len(s), segment_len):
    segment = s[pos:pos+segment_len]
    results = model(segment, is_final=False) 
    # sometimes includes intermediate result for longer input
results = model(torch.empty(0), is_final=True)
```
See [pyadintool](https://github.com/ouktlab/pyadintool) ASR example for real-time streaming ASR. 

## Available Pre-trained Models
Some models are available at huggingface under cc-by-nc-4.0 license. 

### Assumptions
- Number of speakers: one
- Audio file format: monaural 16kHz sampling
  - Raw uncompressed audio is better
  - Up/Down-sampling is required before recognition  
- Pre-processing: voice activity deteciton (VAD) is necessary
  - Non-speech section may affect the performance
  - Other pre-processings, such as speech enhancement, sound source separation, may degrade the performance of some models


### Kanji-Katakana-Hiragana Models
These models are used to estimate Japanese characters from speech signal.
```
あらゆる現実をすべて自分の方へねじ曲げたのだ
```

#### Batch
- [ouktlab/espnet_robustcsj_asr_train_asr_transformer_lm_rnn](https://huggingface.co/ouktlab/espnet_robustcsj_asr_train_asr_transformer_lm_rnn)
  - Multi-conditioned training: clean speech, reverberant speech, and mixture of speech and non-speech signal
#### Streaming
- [ouktlab/espnet_streaming_csj_asr_train_asr_transformer_lm_rnn](https://huggingface.co/ouktlab/espnet_streaming_csj_asr_train_asr_transformer_lm_rnn)
- [ouktlab/espnet_streaming_robustcsj_asr_train_asr_transformer_lm_rnn](https://huggingface.co/ouktlab/espnet_streaming_robustcsj_asr_train_asr_transformer_lm_rnn)
  - Multi-conditioned training: clean speech, reverberant speech, and mixture of speech and non-speech signal
- [ouktlab/espnet_streaming_robustcsj_csjbccwj-v01_asr_train_asr_transformer_lm_rnn](https://huggingface.co/ouktlab/espnet_streaming_robustcsj_csjbccwj-v01_asr_train_asr_transformer_lm_rnn)
  - RNN-LM trained by CSJ and BCCWJ corpus

### Katakana Models
These models are used to estimate Japanese Katakana characters (syllable/pronunciation symbols) from speech signal. The "Katakana" transcription used in training is based on notion of pronunciation. 
- ヲ, ヘ and ヅ are converted into オ, エ and ズ due to their pronuciation. 
- Some vowels are converted into a long vowel: ホウ -> ホー. 
```
アラユルゲンジツオスベテジブンノホーエネジマゲタノダ
```
#### Batch
- [ouktlab/espnet_katakana_csj_asr_train_asr_transformer_lm_rnn](https://huggingface.co/ouktlab/espnet_katakana_csj_asr_train_asr_transformer_lm_rnn)
- [ouktlab/espnet_katakana_robustcorpus10_asr_train_asr_transformer_lm_rnn](https://huggingface.co/ouktlab/espnet_katakana_robustcorpus10_asr_train_asr_transformer_lm_rnn)

#### Streaming
- [ouktlab/espnet_streaming_katakana_csj_asr_train_asr_transformer_lm_rnn](https://huggingface.co/ouktlab/espnet_streaming_katakana_csj_asr_train_asr_transformer_lm_rnn)
- [ouktlab/espnet_streaming_katakana_robustcsj_asr_train_asr_transformer_lm_rnn](https://huggingface.co/ouktlab/espnet_streaming_katakana_robustcsj_asr_train_asr_transformer_lm_rnn)
- [ouktlab/espnet_streaming_katakana_robustcorpus10_asr_train_asr_transformer_lm_rnn](https://huggingface.co/ouktlab/espnet_streaming_katakana_robustcorpus10_asr_train_asr_transformer_lm_rnn)


## Others
### Download Model
We can download models from huggingface by git command. Use git lfs. 
- Install git lfs
```
sudo apt install git-lfs
git lfs install
```
- Clone repository
```
git clone https://huggingface.co/ouktlab/espnet_katakana_csj_asr_train_asr_transformer_lm_rnn
```

### Switch to Different Models
#### Example: Downloaded Model
We can use our own asr and lm models by specifying their path directly.
```
# Load model
basepath = 'espnet_katakana_csj_asr_train_asr_transformer_lm_rnn/exp/'
asr_base = f'{basepath}/asr_train_asr_transformer_ja_raw_jp_char_sp/'
lm_base = f'{basepath}/lm_train_lm_ja_char/'
    
model = Speech2Text.from_pretrained(
    asr_train_config=f'{asr_base}/config.yaml',
    asr_model_file=f'{asr_base}/valid.acc.ave_10best.pth',
    lm_train_config=f'{lm_base}/config.yaml',
    lm_file=f'{lm_base}/valid.loss.ave.pth',
    device=args.device,
    token_type='char',
    bpemodel=None,
    maxlenratio=0.0,
    minlenratio=0.0,
    beam_size=args.beam_size,
    nbest=args.nbest,
    ctc_weight=args.ctc_weight,
    lm_weight=args.lm_weight,
    penalty=0.0
)
```

We must check and change the path of the "stats file" in the asr "config.yaml" according to your environment.
The default setting may be the following.
```
normalize_conf:
    stats_file: exp/asr_stats_raw_jp_char_sp/train/feats_stats.npz
```
Because this represents a relative path from current directory, we need to change it for a different directory. 
```
normalize_conf:
    stats_file: espnet_katakana_csj_asr_train_asr_transformer_lm_rnn/exp/asr_stats_raw_jp_char_sp/train/feats_stats.npz
```

#### Your ESPnet Language Model
Just change the path of "lm_train_config" and "lm_file". It is better to change language model if the application domain is specific. The common "token list" between ASR and LM is assumed.  
```
# Load model
basepath = 'espnet_katakana_csj_asr_train_asr_transformer_lm_rnn/exp/'
asr_base = f'{basepath}/asr_train_asr_transformer_ja_raw_jp_char_sp/'
lm_train_config =     # set path of your configuration file
lm_file =             # set path of your parameter file of lm

model = Speech2Text.from_pretrained(
    asr_train_config=f'{asr_base}/config.yaml',
    asr_model_file=f'{asr_base}/valid.acc.ave_10best.pth',
    lm_train_config=lm_train_config,
    lm_file=lm_file,
    device=args.device,
    token_type='char',
    bpemodel=None,
    maxlenratio=0.0,
    minlenratio=0.0,
    beam_size=args.beam_size,
    nbest=args.nbest,
    ctc_weight=args.ctc_weight,
    lm_weight=args.lm_weight,
    penalty=0.0
)
```
