# ESPnet ASR models
This repository provides pretrained ESPnet ASR models in **"Japanese"** and some examples. 
These models are developed mainly for academic research and Japanese ASR baselines. 
Model parameters are available at [our model hub](https://huggingface.co/ouktlab). 

Please see [extrakit](https://github.com/ouktlab/espnet_asr_extrakit) and [pyadintool](https://github.com/ouktlab/pyadintool) if you are interested in other example codes, such as fine-tuning.

## Features
* Support Japanese models trained with **human-annotated (transciption) corpora**
* Support noise-robust models for raw recoding data (not for audio data of movie file, and compressed audio data)
* Support batch and streaming models (ContextualBlockTransformer for streaming)
* Support Kanji-Katakana-Hiragana and Katakana (Syllable-like) character ASR models
* Support Syllable (Katakana)-to-Character translation (SCT) models (update 2025/8)
* Recognition of fillers and hesitations (deletion error rate is basically small)

## Requirements
Python and ESPnet are required.
- espnet
- torchaudio
- espnet_model_zoo

If you use language model, transformers library is also required in addition to them.
- transformers
- soxr

Python3.10 is suitable for installation of ESPnet (as of 2024/8). 

## How to use
### Set up ESPnet
```
python3 -m venv venv
. venv/bin/activate
python3 -m pip install espnet torchaudio
python3 -m pip install -U espnet_model_zoo
```
If you want to try language models, please install "transformers".
```
python3 -m venv venv
. venv/bin/activate
python3 -m pip install espnet torchaudio transformers soxr
python3 -m pip install -U espnet_model_zoo
```


If you want to setup automatically, run the shell script "setup.sh".
This scripts creates virtual environment and install espnet.
```
sh setup.sh
```

### Case: Normal ASR (batch)
Use "batch.py" for batch processing (start to recognize after a whole input signal is given).
```
$ . venv/bin/activate             # only once
(venv) $ python3 batch.py sample.wav
Result: あらゆる現実を全て自分の方へねじ曲げたのだ
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
Use "streaming.py" for "low-latency" streaming processing. The total processing cost of it may be larger than that of batch processing.
This model may be suitable for spoken dialogue system.
```
$ . venv/bin/activate            # only once
(venv) $ python3 streaming.py sample.wav
```

Bacause "from_pretrained" method has not been implemented in "Speech2TextStreaming" class yet, its wrapper class is defined and used in our example code. 
The following is an example code of streaming recognition.
```
args = usage()  # get parameter settings. This needs to be defined. 

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
    penalty=0.0,
    disable_repetition_detection=True,
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

Please set an appropriate "beam_size" to reduce the latency time because the default beam_size 40 may be large for real-time recognition. 

**Note that the option "disable_repetition_detection" is required to obtain intermediate recognition results.**


See [pyadintool](https://github.com/ouktlab/pyadintool) ASR example for real-time streaming ASR. 

### Case: Syllable (Katakana)-ASR and Syllable-to-Character Translation
Use "sylasr_sct.py" for batch processing using Katakana(syllable)-based ASR and syllable-to-character translation models.
```
$ . venv/bin/activate             # only once
(venv) $ python3 sylasr_sct.py sample.wav
-- Katakana-ASR result --
…アラユルゲンジツオスベテジブンノホーエネジマゲタノダ…
-- SCT result --
…あらゆる現実を全て自分の方へねじ曲げたのだ…
```

```
import torchaudio
from espnet2.bin.asr_inference import Speech2Text
from transformers import AutoTokenizer, T5ForConditionalGeneration

###
args = usage() # get parameter settings. This needs to be defined. 

###
asr_path = 'ouktlab/espnet_asr-ja-kc_am-transformer-robustcorpus10_lm-transformer-corpus10-bccwj-wiki40b'
tokenizer_path = 'ouktlab/character_tokenizer_jis_v2'
sct_path = 'ouktlab/t5_sct-jis-v2_corpus10-bccwj-wiki40b_mask-1.00'

###
asr_model = Speech2Text.from_pretrained(
      asr_path,
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

tokenizer = AutoTokenizer.from_pretrained(tokenizer_path,
                                          trust_remote_code=True)
sct_model = T5ForConditionalGeneration.from_pretrained(sct_path)
sct_model.eval()

###
s, fs = torchaudio.load(args.filename)
s = s.squeeze()
results = asr_model(s)

input_text = [results[0][0]]
inputs = tokenizer(input_text, return_tensors="pt", padding=True)
outputs = sct_model.generate(input_ids=inputs.input_ids,
                             attention_mask=inputs.attention_mask,
                             max_length=len(input_text[0])+32,
                             return_dict_in_generate=True,
                             num_beams=15,
                             length_penalty=0.0,
                             do_sample=False, output_logits=False,
                             output_scores=True)

###
print('-- Katakana-ASR result --')
print(input_text[0])
print('-- SCT result --')
for i, output_ids in enumerate(outputs['sequences']):
  print(tokenizer.decode(output_ids, skip_special_tokens=True))  
```


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

Numbers are represented by Chinese numerals.  
Audio and transcription with ID D*  in CSJ corpus were excluded from training data (following the CSJ recipe of ESPnet).

#### Batch
- [ouktlab/espnet_csj_asr_train_asr_transformer_lm_rnn](https://huggingface.co/ouktlab/espnet_csj_asr_train_asr_transformer_lm_rnn)
  - model: transformer ASR + RNN LM.
  - audio: [CSJ](https://clrd.ninjal.ac.jp/csj/) clean speech
  - text: CSJ transcription
- [ouktlab/espnet_robustcsj_asr_train_asr_transformer_lm_rnn](https://huggingface.co/ouktlab/espnet_robustcsj_asr_train_asr_transformer_lm_rnn)
  - model: transformer ASR + RNN LM.
  - audio: [CSJ](https://clrd.ninjal.ac.jp/csj/) clean speech + augmented speech -- reverberant speech, and mixture of speech and non-speech signal
  - text: CSJ transcription
- [ouktlab/espnet_asr-ja-mc_am-transformer-robustcorpus10_lm-transformer-corpus10-bccwj-wiki40b](https://huggingface.co/ouktlab/espnet_asr-ja-mc_am-transformer-robustcorpus10_lm-transformer-corpus10-bccwj-wiki40b)
  - model: transformer ASR + transformer LM.
  - audio: 10 corpora with transcription ([CSJ](https://clrd.ninjal.ac.jp/csj/), [S-JNAS](https://research.nii.ac.jp/src/S-JNAS.html), [TMW](https://research.nii.ac.jp/src/TMW.html), [JEIDA-JCSD](https://research.nii.ac.jp/src/JEIDA-JCSD.html),[ETL-WD](https://research.nii.ac.jp/src/ETL-WD.html), [RIKEN-DLG](https://research.nii.ac.jp/src/RIKEN-DLG.html), [APP, APPDIC](https://www.atr-p.com/products/sdb.html#MS), [SLC-3](https://alaginrc.nict.go.jp/slc-outline.html#3), [JVS](https://sites.google.com/site/shinnosuketakamichi/research-topics/jvs_corpus)) + augmented speech
    - age distribution: from children to the elderly
    - augmentation: reverberation, non-speech signal ([MUSAN](https://www.openslr.org/17/), [WHAM!](http://wham.whisper.ai/), ProSoundEffect)
    - text standardization: CSJ transcription rule
  - text: transcription of 10 corpora (CSJ, S-JNAS, TMW, JEIDA-JCSD, ETL-WD, RIKEN-DLG, APP, APPDIC, SLC-3, JVS), bccwj, wiki40b-ja, wikipedia-title (2024/8/23)
    - text standardization:  CSJ transcription rule with best effort
  - recommended setting of CTC and language model weights: (0.21, 0.30) . Default setting is not the best.

#### Streaming
- [ouktlab/espnet_streaming_csj_asr_train_asr_transformer_lm_rnn](https://huggingface.co/ouktlab/espnet_streaming_csj_asr_train_asr_transformer_lm_rnn)
  - model: contextual-block-transformer ASR + RNN LM 
  - audio: [CSJ](https://clrd.ninjal.ac.jp/csj/) clean speech
  - text: CSJ transcription
- [ouktlab/espnet_streaming_robustcsj_asr_train_asr_transformer_lm_rnn](https://huggingface.co/ouktlab/espnet_streaming_robustcsj_asr_train_asr_transformer_lm_rnn)
  - model: contextual-block-transformer ASR + RNN LM
  - audio: [CSJ](https://clrd.ninjal.ac.jp/csj/) clean speech + augmented speech -- reverberant speech, and mixture of speech and non-speech signal
  - text: CSJ transcription
- [ouktlab/espnet_streaming_robustcsj_csjbccwj-v01_asr_train_asr_transformer_lm_rnn](https://huggingface.co/ouktlab/espnet_streaming_robustcsj_csjbccwj-v01_asr_train_asr_transformer_lm_rnn)
  - model: contextual-block-transformer ASR + RNN LM
  - audio: [CSJ](https://clrd.ninjal.ac.jp/csj/) clean speech + augmented speech -- reverberant speech, and mixture of speech and non-speech signal
  - text: CSJ transcription + BCCWJ
- [ouktlab/espnet_asr-ja-mc-stream_am-transformer-robustcorpus10_lm-transformer-corpus10-bccwj-wiki40b](https://huggingface.co/ouktlab/espnet_asr-ja-mc-stream_am-transformer-robustcorpus10_lm-transformer-corpus10-bccwj-wiki40b)
  - model: contextual-block-transformer ASR + transformer LM.
  - audio: 10 corpora with transcription ([CSJ](https://clrd.ninjal.ac.jp/csj/), [S-JNAS](https://research.nii.ac.jp/src/S-JNAS.html), [TMW](https://research.nii.ac.jp/src/TMW.html), [JEIDA-JCSD](https://research.nii.ac.jp/src/JEIDA-JCSD.html),[ETL-WD](https://research.nii.ac.jp/src/ETL-WD.html), [RIKEN-DLG](https://research.nii.ac.jp/src/RIKEN-DLG.html), [APP, APPDIC](https://www.atr-p.com/products/sdb.html#MS), [SLC-3](https://alaginrc.nict.go.jp/slc-outline.html#3), [JVS](https://sites.google.com/site/shinnosuketakamichi/research-topics/jvs_corpus)) + augmented speech
    - age distribution: from children to the elderly
    - augmentation: reverberation, non-speech signal ([MUSAN](https://www.openslr.org/17/), [WHAM!](http://wham.whisper.ai/), ProSoundEffect)
    - text standardization: CSJ transcription rule
  - text: transcription of 10 corpora (CSJ, S-JNAS, TMW, JEIDA-JCSD, ETL-WD, RIKEN-DLG, APP, APPDIC, SLC-3, JVS), bccwj, wiki40b-ja, wikipedia-title (2024/8/23)
    - text standardization:  CSJ transcription rule with best effort
  - recommended setting of CTC and language model weights: (0.21, 0.30) . Default setting is not the best.

### Katakana Models
These models are used to estimate Japanese Katakana characters (syllable/pronunciation symbols) from speech signal. The "Katakana" transcription used in training is based on notion of pronunciation. 
- ヲ, ヘ and ヅ are converted into オ, エ and ズ due to their pronuciation. 
- Some vowels are converted into a long vowel: ホウ -> ホー. 
```
アラユルゲンジツオスベテジブンノホーエネジマゲタノダ
```

Audio and transcription with ID D* in CSJ corpus were excluded from training data (following the CSJ recipe of ESPnet).

#### Batch
- [ouktlab/espnet_katakana_csj_asr_train_asr_transformer_lm_rnn](https://huggingface.co/ouktlab/espnet_katakana_csj_asr_train_asr_transformer_lm_rnn)
  - model: transformer ASR and RNN LM.
  - audio: [CSJ](https://clrd.ninjal.ac.jp/csj/) clean speech
  - text: CSJ transcription
- [ouktlab/espnet_katakana_robustcorpus10_asr_train_asr_transformer_lm_rnn](https://huggingface.co/ouktlab/espnet_katakana_robustcorpus10_asr_train_asr_transformer_lm_rnn)
  - model: transformer ASR and RNN LM.
- [ouktlab/espnet_asr-ja-kc_am-transformer-robustcorpus10_lm-transformer-corpus10-bccwj-wiki40b](https://huggingface.co/ouktlab/espnet_asr-ja-kc_am-transformer-robustcorpus10_lm-transformer-corpus10-bccwj-wiki40b)
  - model: transformer ASR and transformer LM.
  - audio: 10 corpora with transcription ([CSJ](https://clrd.ninjal.ac.jp/csj/), [S-JNAS](https://research.nii.ac.jp/src/S-JNAS.html), [TMW](https://research.nii.ac.jp/src/TMW.html), [JEIDA-JCSD](https://research.nii.ac.jp/src/JEIDA-JCSD.html),[ETL-WD](https://research.nii.ac.jp/src/ETL-WD.html), [RIKEN-DLG](https://research.nii.ac.jp/src/RIKEN-DLG.html), [APP, APPDIC](https://www.atr-p.com/products/sdb.html#MS), [SLC-3](https://alaginrc.nict.go.jp/slc-outline.html#3), [JVS](https://sites.google.com/site/shinnosuketakamichi/research-topics/jvs_corpus)) + augmented speech
    - age distribution: from children to the elderly
    - augmentation: reverberation, non-speech signal ([MUSAN](https://www.openslr.org/17/), [WHAM!](http://wham.whisper.ai/), ProSoundEffect)
    - text standardization: CSJ transcription rule
  - text: transcription of 10 corpora (CSJ, S-JNAS, TMW, JEIDA-JCSD, ETL-WD, RIKEN-DLG, APP, APPDIC, SLC-3, JVS), bccwj, wiki40b-ja, wikipedia-title (2024/8/23)
    - text standardization:  CSJ transcription rule with best effort
  - recommended setting of CTC and language model: (0.21, 0.30) or (0.19, 0.35) . Default setting is not the best.

#### Streaming
- [ouktlab/espnet_streaming_katakana_csj_asr_train_asr_transformer_lm_rnn](https://huggingface.co/ouktlab/espnet_streaming_katakana_csj_asr_train_asr_transformer_lm_rnn)
  - model: contextual-block-transformer ASR + RNN LM 
  - audio: [CSJ](https://clrd.ninjal.ac.jp/csj/) clean speech
  - text: CSJ transcription
- [ouktlab/espnet_streaming_katakana_robustcsj_asr_train_asr_transformer_lm_rnn](https://huggingface.co/ouktlab/espnet_streaming_katakana_robustcsj_asr_train_asr_transformer_lm_rnn)
  - model: contextual-block-transformer ASR + RNN LM 
  - audio: [CSJ](https://clrd.ninjal.ac.jp/csj/) clean speech + augmented speech -- reverberant speech, and mixture of speech and non-speech signal
  - text: CSJ transcription
- [ouktlab/espnet_asr-ja-kc-stream_am-transformer-robustcorpus10_lm-transformer-corpus10-bccwj-wiki40b](https://huggingface.co/ouktlab/espnet_asr-ja-kc-stream_am-transformer-robustcorpus10_lm-transformer-corpus10-bccwj-wiki40b)
  - model: contextual-block-transformer ASR + transformer LM
  - audio: 10 corpora with transcription ([CSJ](https://clrd.ninjal.ac.jp/csj/), [S-JNAS](https://research.nii.ac.jp/src/S-JNAS.html), [TMW](https://research.nii.ac.jp/src/TMW.html), [JEIDA-JCSD](https://research.nii.ac.jp/src/JEIDA-JCSD.html),[ETL-WD](https://research.nii.ac.jp/src/ETL-WD.html), [RIKEN-DLG](https://research.nii.ac.jp/src/RIKEN-DLG.html), [APP, APPDIC](https://www.atr-p.com/products/sdb.html#MS), [SLC-3](https://alaginrc.nict.go.jp/slc-outline.html#3), [JVS](https://sites.google.com/site/shinnosuketakamichi/research-topics/jvs_corpus)) + augmented speech
    - age distribution: from children to the elderly
    - augmentation: reverberation, non-speech signal ([MUSAN](https://www.openslr.org/17/), [WHAM!](http://wham.whisper.ai/), ProSoundEffect)
    - text standardization: CSJ transcription rule
  - text: transcription of 10 corpora (CSJ, S-JNAS, TMW, JEIDA-JCSD, ETL-WD, RIKEN-DLG, APP, APPDIC, SLC-3, JVS), bccwj, wiki40b-ja, wikipedia-title (2024/8/23)
    - text standardization:  CSJ transcription rule with best effort
  - recommended setting of CTC and language model: (0.21, 0.30) or (0.19, 0.35) . Default setting is not the best.

### Syllable-to-Character Translation (SCT) Models 
These syllable-to-character translation (SCT) models are not ESPnet models, but they can be used for Japanese ASR based on syllable (Katakana)-ASR and KanaKanji-Convresion. The following is an example of such ASR framework.
 - Katakana-ASR: signal -> アラユルゲンジツオ
 - SCT: アラユルゲンジツオ -> あらゆる現実を
 
SCT models estimate Japanese characters (Kanji, Hiragana, Katakan) from Japanese Katakana characters (syllable/pronunciation symbols). 
ASR is achieved by combining Katakana- (syllable-)ASR models with SCT models. In this framework, we can design and develop each model independently, which will improve model portability.

The training of SCT model requires only parallel text (pairs of Katakana/Syllable and Kanji-Katakana-Hiragana sequences), in other words, audio data are not required. 
- アラユルゲンジツオ  あらゆる現実を

We can do fine-tuning or scrach-training of these models using other models by "transformers" library. 

#### Tokenizer

- [ouktlab/character_tokenizer_jis_v1](https://huggingface.co/ouktlab/character_tokenizer_jis_v1)
  - unit: character
  - vocabulary set: JIS X 0213 (Japanese Industrial Standard for coded character sets)
- [ouktlab/character_tokenizer_jis_v2](https://huggingface.co/ouktlab/character_tokenizer_jis_v2)
  - unit: character
  - vocabulary set: JIS X 0213 (Japanese Industrial Standard for coded character sets) -- Zenkaku period (for decimal point) is added correctly. 

#### T5 for Conditional Generation
These preliminary models use T5 model for conditional generation. Note that architechtures and tokenizers have not been optimized yet.   

Please note that our pause (non-speech) symbol, "…", also plays a role of separator among sentences or words segmented by VAD or hands. 

- [ouktlab/t5_sct-jis-v1_corpus10-bccwj-wiki40b_std](https://huggingface.co/ouktlab/t5_sct-jis-v1_corpus10-bccwj-wiki40b_std)
  - tokenizer: character jis v1
  - text: 10 corpora + bccwj + wiki40b-ja + wikipedia-title (2024/8/23)
  - estimation of word pronunciation for text only data set: mecab with unidic and NEologd dictionaries
- [ouktlab/t5_sct-jis-v1_corpus10-bccwj-wiki40b_mask-1.00](https://huggingface.co/ouktlab/t5_sct-jis-v1_corpus10-bccwj-wiki40b_mask-1.00)
  - tokenizer: character jis v1
  - text: 10 corpora + bccwj + wiki40b-ja + wikipedia-title (2024/8/23)
  - augmentation: syllable-ASR error simulation using MASK token
  - estimation of word pronunciation for text only data set: mecab with unidic and NEologd dictionaries
- [ouktlab/t5_sct-jis-v2_corpus10-bccwj-wiki40b_mask-1.00](https://huggingface.co/ouktlab/t5_sct-jis-v2_corpus10-bccwj-wiki40b_mask-1.00)
  - tokenizer: character jis v2
  - text: 10 corpora + bccwj + wiki40b-ja + wikipedia-title (2024/8/23)
  - augmentation: syllable-ASR error simulation using MASK token
  - estimation of word pronunciation for text only data set: mecab with unidic and NEologd dictionaries

#### Usage and Example
We can use the SCT models by using AutoTokenizer and ConditionalGeneration classed. Please note that "trust_remote_code" option is required because our tokenizer is customized.  

The following is an example code from "sct.py". Beam search is applied to estimate a high score hypothesis. The "model.eval()" and "do_sample=False" are necessary for reproducibility (non stochastic search).
```
from transformers import AutoTokenizer, T5ForConditionalGeneration

###
tokenizer_path = 'ouktlab/character_tokenizer_jis_v2'
model_path = 'ouktlab/t5_sct-jis-v2_corpus10-bccwj-wiki40b_mask-1.00'

###
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, 
                                          trust_remote_code=True)
model = T5ForConditionalGeneration.from_pretrained(model_path)
model.eval()

###
input_text = ['ニセンジューハチネンノイチガツカラ…ホーソーサレルヨーデスヨ']
inputs = tokenizer(input_text, return_tensors="pt", padding=True)
outputs = model.generate(input_ids=inputs.input_ids,
                         attention_mask=inputs.attention_mask,
                         max_length=len(input_text[0])+32,
                         return_dict_in_generate=True,
                         num_beams=15,
                         length_penalty=0.0,
                         do_sample=False, output_logits=False, 
                         output_scores=True)
scores = outputs.sequences_scores
    
for i, output_ids in enumerate(outputs['sequences']):
    print(f'{scores[i]:.2e}', 
          tokenizer.decode(output_ids, skip_special_tokens=True))
```
Please run "sct.py", and you will get the translation result. 
```
(venv) $ python3 sct.py
-8.95e-02 二千十八年の一月から…放送されるようですよ
```


### Language Model for verification of recognition result
#### GPTNeoX
- [gptneox_ja-neox-small_corpus10-bccwj-wiki40b](https://huggingface.co/ouktlab/gptneox_ja-neox-small_corpus10-bccwj-wiki40b)
  - tokenizer: rinna/japanese-gpt-neox-small
  - text: corpus10 + bccwj + wiki40b-ja + wikipedia-title (2024/8/23)

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

### Modification for streaming ASR
It is better to change the default parameters of ContextualBlockTransformer 
because some of our models are slightly different from defaults of streaming ASR.  
We may be able to get intermediate results more frequently by changing these parameters. 

Our settings (*corpus10* models)
```
  block_size: int = 20,
  hop_size: int = 8,
  look_ahead: int = 8,
```
Default settings
```
  block_size: int = 40,
  hop_size: int = 16,
  look_ahead: int = 16,
```

If you want to adjust these parameters, please modify the source code of ESPnet as follows.
1. Remove comment-outs: asr_inference_streaming.py, line 117
```
  look_ahead = asr_train_args.encoder_conf['look_ahead']
  hop_size   = asr_train_args.encoder_conf['hop_size']
  block_size = asr_train_args.encoder_conf['block_size']
```
2. Add default parameters: asr_inference_streaming.py, line 132-134
```
  beam_search = BatchBeamSearchOnline(
    beam_size=beam_size,
    weights=weights,
    scorers=scorers,
    sos=asr_model.sos,
    eos=asr_model.eos,
    vocab_size=len(token_list),
    token_list=token_list,
    pre_beam_score_key=None if ctc_weight == 1.0 else "full",
    block_size=block_size, # added
    hop_size=hop_size,     # added
    look_ahead=look_ahead, # added
    disable_repetition_detection=disable_repetition_detection,
    decoder_text_length_limit=decoder_text_length_limit,
    encoded_feat_length_limit=encoded_feat_length_limit,
  )
```

# Disclaimer
We shall not be liable for any trouble, loss and damage caused by the use of models, codes, and website.

# Citations
### Syllable-based ASR and SCT approach
- ASR (both of character and syllable) and SCT models using "10 corpora" data set
- ASR performance comparison
```
@inproceedings {rtakeda2025:apsipa,
  author={Ryu Takeda and Kazunori Komatani},
  title={Reducing Orthographic Dependency on Paired Data by Probabilistic Integration via Syllabogram for Japanese Dialogue Speech Recognition},
  year={2025},
  booktitle={Proceedings of Asia-Pacific Signal and Information Processing Association Annual Summit and Conference (APSIPA ASC) (to appear)},
}
```