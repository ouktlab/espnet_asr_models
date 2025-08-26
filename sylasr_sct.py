import torchaudio
from espnet2.bin.asr_inference import Speech2Text
from transformers import AutoTokenizer, T5ForConditionalGeneration

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

###
def main():
    ###
    args = usage()

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

if __name__ == '__main__':
    main()
