import torch
import torchaudio
import time
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple, Union
from espnet2.bin.asr_inference_streaming import Speech2TextStreaming

'''
Streaming Interface for implementing 'from_pretrained' method
which has not been implemented in ESPnet (pip version).
Note that the original code of 'from_pretrained' is implemented for Speech2Text class in ESPnet.
line 670 @ https://github.com/espnet/espnet/blob/master/espnet2/bin/asr_inference.py
(ESPnet https://github.com/espnet/espnet, Apache License by Shinji Watanabe)
We used it for Speech2TextStreaming class here.
'''
class Speech2TextStreamingInterface(Speech2TextStreaming):
    def __init__(self, **kwargs):
        super.__init__(kwargs)
        
    @staticmethod
    def from_pretrained(
            model_tag: Optional[str] = None,
            **kwargs: Optional[Any],
    ):
        if model_tag is not None:
            try:
                from espnet_model_zoo.downloader import ModelDownloader
            except ImportError:
                print(
                    "`espnet_model_zoo` is not installed. "
                    "Please install via `pip install -U espnet_model_zoo`."
                )
                raise
            d = ModelDownloader()
            kwargs.update(**d.download_and_unpack(model_tag))
        return Speech2TextStreaming(**kwargs)
        
def usage():
    """
    return
        args: argparse.Namespace
    """
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('filename', type=str, help='audio filename')
    parser.add_argument('--device', type=str, help='cpu or cuda', default='cpu')
    parser.add_argument('--nbest', type=int, help='nbest', default=40)
    parser.add_argument('--beam_size', type=int, help='beam size', default=40)
    parser.add_argument('--ctc_weight', type=float, help='nbest', default=0.3)
    parser.add_argument('--lm_weight', type=float, help='lm weight', default=0.1)
    parser.add_argument('--penalty', type=float, help='lm weight', default=0.0)

    args = parser.parse_args()
    return args

def main():
    args = usage()

    # Repository URL. Change here if you want to change ESPnet model. 
    hfrepo = 'ouktlab/espnet_streaming_robustcsj_asr_train_asr_transformer_lm_rnn'
    #hfrepo = 'ouktlab/espnet_streaming_katakana_csj_asr_train_asr_transformer_lm_rnn'

    # Load model
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

    #
    s, fs = torchaudio.load(args.filename)
    s = s.squeeze()
    
    print(f'[LOG]: start recognition')
    
    segment_len = 1600
    for pos in range(0, len(s), segment_len):
        segment = s[pos:pos+segment_len]
        results = model(segment, is_final=False)
    results = model(torch.empty(0), is_final=True)
    
    print(f'Result: {results[0][0]}')
    print(f'  score: {results[0][3][1].item():.2f}')

if __name__ == '__main__':
    main()
