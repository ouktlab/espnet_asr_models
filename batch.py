import torchaudio
import time
from espnet2.bin.asr_inference import Speech2Text

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
    hfrepo = 'ouktlab/espnet_robustcsj_asr_train_asr_transformer_lm_rnn'
    #hfrepo = 'ouktlab/espnet_katakana_csj_asr_train_asr_transformer_lm_rnn'

    # Load model
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

    #
    s, fs = torchaudio.load(args.filename)
    s = s.squeeze()
    
    print(f'[LOG]: start recognition')
    start_time = time.perf_counter()
    results = model(s)
    perftime = time.perf_counter() - start_time
    print(f'Result: {results[0][0]}')
    print(f'  score: {results[0][3][1].item():.2f}, RTF: {perftime/(len(s)/fs):.2f}')
    

if __name__ == '__main__':
    main()
