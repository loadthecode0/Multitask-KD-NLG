import sys

# Remove all entries that point to the MonteCLoRA version of peft
sys.path = [p for p in sys.path if "MonteCLoRA/peft" not in p]

# Sanity check
import peft
print("Using PEFT from:", peft.__file__)


import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import get_peft_model, LoraConfig, TaskType
from trl import SFTTrainer
from datasets import load_dataset

os.environ["WANDB_PROJECT"] = "ell884-proj"
from huggingface_hub import login
login("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")
# import sys
# sys.path = [p for p in sys.path if "MonteCLoRA/peft" not in p]

model_name = "Meta-Llama-3.1-8B/snapshots/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b/"  # Replace with your desired HF model

# Load tokenizer and model in full precision
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    token='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'
)
tokenizer.chat_template = None

# Apply LoRA
peft_config = LoraConfig(
    r=32,
    lora_alpha=64,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)
model = get_peft_model(model, peft_config)
tokenizer.pad_token = tokenizer.eos_token

# Prompt formatting
EOS_TOKEN = tokenizer.eos_token
def formatting_prompts_func(example):
    instruction = example["instruction"]
    input_text = example.get("input", "")
    output_text = example["output"]

    if input_text:
        prompt = f"""### Instruction:
{instruction}

### Input:
{input_text}

### Response:
{output_text}""" + EOS_TOKEN
    else:
        prompt = f"""### Instruction:
{instruction}

### Response:
{output_text}""" + EOS_TOKEN

    # prompt = example["text"]
    tokens = tokenizer(prompt, padding="max_length", truncation=True, max_length=1024)
    tokens["labels"] = tokens["input_ids"].copy()  # so model can compute loss
    return tokens
    # return {"text": prompt}
    # return prompt

# Load and prepare dataset
dataset = load_dataset("json", data_files="instruction_combined_1.jsonl")["train"]
dataset = dataset.shuffle(seed=120).select(range(int(0.05 * len(dataset)))).train_test_split(test_size=0.1, seed=121)
print(dataset)
train_data = dataset["train"].map(formatting_prompts_func)
val_data   = dataset["test"].map(formatting_prompts_func)

print(train_data)
print(train_data[0])

# Training
trainer = Trainer(
    model=model,
    # processing_class=tokenizer,
    # train_dataset=train_data,
    # eval_dataset=val_data,
    # dataset_text_field="text",
    # max_seq_length=1024,
    tokenizer=tokenizer,
    train_dataset=train_data,
    eval_dataset=val_data,
    # dataset_text_field="text",
    # formatting_func=lambda x: x['text'], 
    args=TrainingArguments(
        # dataset_num_proc=4,
        # packing=False,
        label_names = ["labels"],
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        warmup_steps=50,
        num_train_epochs=10,
        learning_rate=1e-4,
        fp16=torch.cuda.is_available(),
        logging_steps=50,
        eval_strategy='steps',
        eval_steps=500,
        save_total_limit=5,
        save_strategy='steps',
        save_steps=500,
        load_best_model_at_end=True,
        metric_for_best_model='eval_loss',
        greater_is_better=False,
        optim="adamw_torch_fused",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs",
        report_to="wandb",
        run_name='teacher-ft-5'
    ),
)

# GPU info
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")

# Train and save
trainer_stats = trainer.train()
model.save_pretrained("teacher_ft_5")
tokenizer.save_pretrained("teacher_ft_5")
