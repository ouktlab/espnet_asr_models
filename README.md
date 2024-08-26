# ESPnet ASR models
This repository provides ESPnet ASR models in Japanese and some examples for usage. These models are developed mainly for academic research. 

## Features
* Support Japanese models
* Support Noise-robust models
* Support batch and streaming models
* Support KanaKanjiMajiri and Katakana (Syllable-like) character models

## How to use
### Set up ESPnet
Use "python3.10" for ESPnet (2024/8).
```
python3 -m pip install espnet torchaudio
python3 -m pip install -U espnet_model_zoo
```

If you want to setup automatically, run the shell script "setup.sh".
This scripts creates virtual environment and install espnet. ls
```
sh setup.sh
```

### Case: Batch ASR
Use "batch.py" for batch processing.
```
. venv/bin/activate             # only once
python3 batch.py sample.wav
```

Loading model is achieved by calling "from_pretrained" method of "Speech2Text" class. 
```
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
```

### Case: Streaming ASR
Use "streaming.py" for low-latency streaming processing. This model may be suitable for spoken dialogue system. GPU processing is recommended. 
```
. venv/bin/activate            # only once
python3 streaming.py sample.wav
```

Bacause "from_pretrained" method has not been implemented in "Speech2TextStreaming" class yet, its wrapper class is defined and used.
```
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
```


## KanaKanjiMajiri-Character Models
Some models are avairable at huggingface under cc-by-nc-4.0 license. 
### Batch
- [ouktlab/espnet_robustcsj_asr_train_asr_transformer_lm_rnn](https://huggingface.co/ouktlab/espnet_robustcsj_asr_train_asr_transformer_lm_rnn)

### Streaming
- [ouktlab/espnet_streaming_csj_asr_train_asr_transformer_lm_rnn](https://huggingface.co/ouktlab/espnet_streaming_csj_asr_train_asr_transformer_lm_rnn)
- [ouktlab/espnet_streaming_robustcsj_asr_train_asr_transformer_lm_rnn](https://huggingface.co/ouktlab/espnet_streaming_robustcsj_asr_train_asr_transformer_lm_rnn)
- [ouktlab/espnet_streaming_robustcsj_csjbccwj-v01_asr_train_asr_transformer_lm_rnn](https://huggingface.co/ouktlab/espnet_streaming_robustcsj_csjbccwj-v01_asr_train_asr_transformer_lm_rnn)

## Katakana-Character Models
### Batch
- [ouktlab/espnet_katakana_csj_asr_train_asr_transformer_lm_rnn](https://huggingface.co/ouktlab/espnet_katakana_csj_asr_train_asr_transformer_lm_rnn)
- [ouktlab/espnet_katakana_robustcorpus10_asr_train_asr_transformer_lm_rnn](https://huggingface.co/ouktlab/espnet_katakana_robustcorpus10_asr_train_asr_transformer_lm_rnn)

### Streaming
- [ouktlab/espnet_streaming_katakana_csj_asr_train_asr_transformer_lm_rnn](https://huggingface.co/ouktlab/espnet_streaming_katakana_csj_asr_train_asr_transformer_lm_rnn)
- [ouktlab/espnet_streaming_katakana_robustcsj_asr_train_asr_transformer_lm_rnn](https://huggingface.co/ouktlab/espnet_streaming_katakana_robustcsj_asr_train_asr_transformer_lm_rnn)
- [ouktlab/espnet_streaming_katakana_robustcorpus10_asr_train_asr_transformer_lm_rnn](https://huggingface.co/ouktlab/espnet_streaming_katakana_robustcorpus10_asr_train_asr_transformer_lm_rnn)

