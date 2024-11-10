from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import numpy as np
from tqdm.auto import tqdm
import gc 

class Evaluation:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
    
    def message_perplexity(self, message):
        encodings = self.tokenizer(message, return_tensors="pt", truncation=True, max_length = 256).to('cuda')
        target_ids = encodings.input_ids.clone()
        outputs = self.model(encodings.input_ids, labels=target_ids)
        neg_log_likelihood = outputs.loss
        return np.exp(neg_log_likelihood.item())
    
    def perplexity(self, messages):
        ppls = []
        for i, m in tqdm(enumerate(messages)):
            message_ppl = self.message_perplexity(m)
            ppls.append(message_ppl)
        
        return np.mean(ppls)