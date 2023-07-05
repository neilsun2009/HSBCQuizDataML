import numpy as np
import os
import math
import pandas as pd
from datasets import Dataset
import argparse
from transformers import (
    AutoTokenizer, DataCollatorForLanguageModeling,
    AutoModelForCausalLM, TrainingArguments, Trainer,
)
from peft import LoraConfig, TaskType, get_peft_model

BLOCK_SIZE = 128

tokenizer = AutoTokenizer.from_pretrained("distilgpt2")

def tokenize(examples):
    return tokenizer([" ".join(x) for x in examples["review"]])

def group_texts(examples):
    # Concatenate all texts.
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
    # customize this part to your needs.
    if total_length >= BLOCK_SIZE:
        total_length = (total_length // BLOCK_SIZE) * BLOCK_SIZE
    # Split by chunks of block_size.
    result = {
        k: [t[i : i + BLOCK_SIZE] for i in range(0, total_length, BLOCK_SIZE)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--comment_path', type=str, required=True, help='input csv file path for all comments')
    parser.add_argument('--output_dir', type=str, required=True, help='output dir for model')
    parser.add_argument('--lr', type=float, default=2e-5, help='learning rate')
    parser.add_argument('--max_line_number', type=int, default=-1, help='max line number to load')
    args = parser.parse_args()

    # prepare dataset
    comment_df = pd.read_csv(args.comment_path, nrows=None if args.max_line_number == -1 else args.max_line_number)
    comment_df = comment_df[['review']]
    ds = Dataset.from_pandas(comment_df)
    ds = ds.train_test_split(test_size=0.2)

    # preprocess
    tokenized_ds = ds.map(
        tokenize,
        batched=True,
        num_proc=4,
        remove_columns=ds["train"].column_names,
    )
    lm_dataset = tokenized_ds.map(
        group_texts, 
        batched=True, 
        num_proc=4,
    )

    tokenizer.pad_token = tokenizer.eos_token
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # lora
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, 
        inference_mode=False, 
        r=8, 
        lora_alpha=32, 
        lora_dropout=0.1,
    )

    # train
    model = AutoModelForCausalLM.from_pretrained("distilgpt2")
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        evaluation_strategy="epoch",
        learning_rate=args.lr,
        weight_decay=0.01,
        push_to_hub=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=lm_dataset["train"],
        eval_dataset=lm_dataset["test"],
        data_collator=data_collator,
    )

    trainer.train()

    # save model
    model.save_pretrained(args.output_dir)

    # evaluate
    eval_results = trainer.evaluate()
    print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")


if __name__ == '__main__':
    main()
