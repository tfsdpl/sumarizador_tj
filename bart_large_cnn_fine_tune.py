from transformers import BartTokenizer, BartForConditionalGeneration, Trainer, TrainingArguments
from datasets import load_dataset
import torch


tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
dataset = load_dataset("c4", "pt", split="train")



def process_data_to_model_inputs(batch):
    inputs = tokenizer(batch["text"], padding="max_length", truncation=True, max_length=1024)
    outputs = tokenizer(batch["summary"], padding="max_length", truncation=True, max_length=128)

    batch["input_ids"] = inputs.input_ids
    batch["attention_mask"] = inputs.attention_mask
    batch["labels"] = outputs.input_ids

    return batch


def fine_tuning():
    tokenized_dataset = dataset.map(process_data_to_model_inputs, batched=True)
    training_args = TrainingArguments(
        output_dir="./results",
        learning_rate=2e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=3,
        weight_decay=0.01,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"]
    )

    trainer.train()
    model.save_pretrained("./bart_large_cnn_finetuned")
