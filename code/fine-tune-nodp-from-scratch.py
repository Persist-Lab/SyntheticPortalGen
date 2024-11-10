# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

'''Train GPT2 model series without DP (w/ parameter-efficient approach LoRA when lora_dim > 0)'''

# Based on code from dp-transformers github https://github.com/microsoft/dp-transformers/tree/main/examples/nlg-reddit/sample-level-dp

import datasets
import dp_transformers
import transformers
import sys, os
import logging
from datasets import Dataset, DatasetDict
from dataclasses import dataclass, field
from dataclasses import dataclass, field, asdict
from peft import get_peft_model, LoraConfig
import pandas as pd
import random

logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    model_name: str = field(default="gpt2", metadata={
        "help": "Model name in HuggingFace, e.g. 'gpt2'"
    })
    sequence_len: int = field(default=128, metadata={
        "help": "Maximum sequence length"
    })
    dataset_path: str = field(default=None, metadata={
        "help": "path to data csv"
    })
    save_model_path: str = field(default=None, metadata={
        "help": "exact save path for trained model"
    })


@dataclass
class LoraArguments:
    enable_lora: bool = field(default=False, metadata={
        "help": "Whether to enable LoRA"
    })
    lora_dim: int = field(default=8, metadata={
        "help": "LoRA dimension"
    })
    lora_alpha: int = field(default=8, metadata={
        "help": "LoRA alpha"
    })
    lora_dropout: float = field(default=0.0, metadata={
        "help": "LoRA dropout"
    })

    def as_peft_config(self) -> LoraConfig:
        if not self.enable_lora:
            raise ValueError("LoRA is not enabled, cannot convert to LoRA config")
        params = asdict(self)
        params.pop("enable_lora")
        params["r"] = params.pop("lora_dim")
        return LoraConfig(**params)


@dataclass
class Arguments:
    train: dp_transformers.TrainingArguments
    model: ModelArguments
    lora: LoraArguments

from transformers import GPT2LMHeadModel, AutoConfig, AutoTokenizer
def main(args: Arguments):
    
   
    print(args.model.dataset_path)
    print(args.model.save_model_path)
    transformers.set_seed(args.train.seed)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = train_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {train_args.local_rank}, device: {train_args.device}, n_gpu: {train_args.n_gpu}, "
        f"distributed training: {bool(train_args.local_rank != -1)}, 16-bits training: {train_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {train_args}")

    # Load model
    
    def create_model(tokenizer):
        
        config = AutoConfig.from_pretrained(
            args.model.model_name,
            vocab_size=len(tokenizer),
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        model = GPT2LMHeadModel(config)
        return model 
    

    # Load data
    df = pd.read_csv(args.model.dataset_path)
    
    dataset = DatasetDict({'train':Dataset.from_pandas(df)})
    # Load tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # model = transformers.AutoModelForCausalLM.from_pretrained(args.model.model_name)
    model = create_model(tokenizer)
    model = model.to(train_args.device)
    print('NUMBER OF MODEL PARAMETERS: ',model.num_parameters())
    
    # Tokenize data
    with train_args.main_process_first(desc="tokenizing dataset"):
        dataset = dataset.map(
            lambda batch: tokenizer(batch['patient_message'], padding="max_length", truncation=True, max_length=args.model.sequence_len),
            batched=True, num_proc=8, desc="tokenizing dataset", remove_columns=dataset.column_names['train']
        )

    if args.lora.enable_lora:
        logger.info("Using LoRA")
        model = get_peft_model(model=model, peft_config=args.lora.as_peft_config())
    else:
        logger.info("Not using LoRA")

    if train_args.local_rank == 0:
        logger.info(f"Total number of parameters of the model: {model.num_parameters(only_trainable=False)}")
        logger.info(f"Fine-tuned number of parameters of the model: {model.num_parameters(only_trainable=True)}")

    model = model.cuda()
    model.train()

    data_collator = dp_transformers.DataCollatorForPrivateCausalLanguageModeling(tokenizer)

    trainer = transformers.Trainer(
        args=train_args,
        model=model,
        train_dataset=dataset['train'],
        data_collator=data_collator
    )

    trainer.train()
    
    model.save_pretrained(args.model.save_model_path)
    tokenizer.save_pretrained(args.model.save_model_path)

if __name__ == "__main__":
    arg_parser = transformers.HfArgumentParser((dp_transformers.TrainingArguments, ModelArguments, LoraArguments))
    train_args, model_args, lora_args = arg_parser.parse_args_into_dataclasses()
    main(Arguments(train=train_args, model=model_args, lora=lora_args))
