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
    print(f'{scores[i]:.2e}', tokenizer.decode(output_ids, skip_special_tokens=True))
