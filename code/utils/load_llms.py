import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline

def load_llm_in_4bit(model_id):



    nf4_config = BitsAndBytesConfig(
       load_in_4bit=True,
       bnb_4bit_quant_type="nf4",
       bnb_4bit_use_double_quant=True,
       bnb_4bit_compute_dtype=torch.bfloat16
    )


    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map='auto',
        quantization_config=nf4_config,
        use_cache=False

    )

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    pipe = pipeline('text-generation', model = model, tokenizer=tokenizer, max_new_tokens = 256, pad_token_id = tokenizer.eos_token_id)
    
    return pipe, tokenizer, model

