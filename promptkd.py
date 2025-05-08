import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import wandb
from peft import PeftModel, PeftConfig, get_peft_model, LoraConfig, TaskType

# Environment setup
os.environ["WANDB_PROJECT"] = "ell884-proj"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Running on device:", device)

# Login for Hugging Face (replace token if needed)
from huggingface_hub import login
login("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")


# CONFIG =================
teacher_model_path = "./outputs/checkpoint-1500"
student_model_path = "./llama-3.2-1B"
# =========================

# Load LoRA teacher model
teacher_config = PeftConfig.from_pretrained(teacher_model_path)
base_teacher_model = AutoModelForCausalLM.from_pretrained(
    teacher_config.base_model_name_or_path,
    torch_dtype=torch.float16
).to(device)

teacher_model = PeftModel.from_pretrained(base_teacher_model, teacher_model_path).to(device)
teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_config.base_model_name_or_path)
teacher_tokenizer.pad_token = teacher_tokenizer.eos_token
teacher_model.eval()

# Load student base model and apply LoRA configuration
student_base_model = AutoModelForCausalLM.from_pretrained(
    student_model_path,
    torch_dtype=torch.float16
).to(device)

# Define LoRA configuration for student
student_lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],  # adjust to model architecture
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

# Apply LoRA to student model
student_model = get_peft_model(student_base_model, student_lora_config).to(device)
student_tokenizer = AutoTokenizer.from_pretrained(student_model_path)
student_tokenizer.pad_token = student_tokenizer.eos_token

# Prompt embedding module
class PromptEmbedding(nn.Module):
    def __init__(self, prompt_length, hidden_size):
        super().__init__()
        self.prompt_embeddings = nn.Parameter(
            torch.randn(prompt_length, hidden_size) * 0.02
        )

    def forward(self, batch_size):
        return self.prompt_embeddings.unsqueeze(0).expand(batch_size, -1, -1)

prompt_length = 20
hidden_size = teacher_model.get_input_embeddings().embedding_dim
prompt_module = PromptEmbedding(prompt_length, hidden_size).to(device)

# Loss functions
def distillation_loss(student_logits, teacher_logits):
    student_log_probs = F.log_softmax(student_logits, dim=-1)
    teacher_probs = F.softmax(teacher_logits, dim=-1)
    return F.kl_div(student_log_probs, teacher_probs, reduction='batchmean', log_target=False)

def supervised_loss(student_logits, labels):
    shift_logits = student_logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
    return loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

def regularization_loss(teacher_logits_with_prompt, teacher_logits_no_prompt):
    log_probs_with_prompt = F.log_softmax(teacher_logits_with_prompt, dim=-1)
    probs_no_prompt = F.softmax(teacher_logits_no_prompt, dim=-1)
    return F.kl_div(log_probs_with_prompt, probs_no_prompt, reduction='batchmean', log_target=False)

# Forward pass with prompt
def teacher_forward_with_prompt(teacher_model, input_ids, attention_mask, prompt_module):
    batch_size = input_ids.size(0)
    inputs_embeds = teacher_model.get_input_embeddings()(input_ids)
    prompt_embeds = prompt_module(batch_size)
    inputs_embeds = torch.cat([prompt_embeds, inputs_embeds], dim=1)

    prompt_attention = torch.ones(batch_size, prompt_embeds.size(1)).to(attention_mask.device)
    attention_mask = torch.cat([prompt_attention, attention_mask], dim=1)

    outputs = teacher_model(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        use_cache=False,
    )
    return outputs.logits

# Optimizers
prompt_optimizer = torch.optim.AdamW(prompt_module.parameters(), lr=5e-5)
student_optimizer = torch.optim.AdamW(student_model.parameters(), lr=5e-5)

# Dataset loading and preprocessing
dataset = load_dataset("json", data_files="instruction_combined_1.jsonl")['train']
EOS_TOKEN = student_tokenizer.eos_token
dataset = dataset.shuffle(seed=120).select(range(int(0.05 * len(dataset))))
split_dataset = dataset.train_test_split(test_size=0.1, seed=121)
train_data, val_data = split_dataset["train"], split_dataset["test"]

def preprocess_function(example):
    prompt = f"""### Instruction:
{example["instruction"]}

### Input:
{example["input"]}

### Response: """ + EOS_TOKEN

    input_enc = student_tokenizer(prompt, truncation=True, padding="max_length", max_length=512, return_tensors="pt")
    output_enc = student_tokenizer(example["output"] + EOS_TOKEN, truncation=True, padding="max_length", max_length=512, return_tensors="pt")

    return {
        "input_ids": input_enc["input_ids"].squeeze(0),
        "attention_mask": input_enc["attention_mask"].squeeze(0),
        "labels": output_enc["input_ids"].squeeze(0),
    }

train_dataset = train_data.map(preprocess_function)
val_dataset = val_data.map(preprocess_function)
train_dataset.set_format(type="torch")
val_dataset.set_format(type="torch")

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=lambda x: {
    "input_ids": torch.stack([item["input_ids"] for item in x]),
    "attention_mask": torch.stack([item["attention_mask"] for item in x]),
    "labels": torch.stack([item["labels"] for item in x]),
})
val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, collate_fn=lambda x: {
    "input_ids": torch.stack([item["input_ids"] for item in x]),
    "attention_mask": torch.stack([item["attention_mask"] for item in x]),
    "labels": torch.stack([item["labels"] for item in x]),
})

# Training loop
total_steps = 10000
current_step = 0
VALIDATION_EVERY_N_STEPS = 200
scaler = GradScaler()
best_val_loss = float('inf')
last_best_checkpoint_folder = None
last_best_prompt_file = None

wandb.init(
    project="promptkd-distillation",
    name="distill-run-1",
    config={
        "epochs": 3,
        "batch_size": 2,
        "learning_rate": 5e-5,
        "prompt_length": prompt_length,
        "student_model": teacher_model_path,
        "teacher_model": student_model_path,
    }
)

for epoch in range(3):
    for batch in train_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        batch_size = input_ids.size(0)

        with autocast():
            teacher_logits_with_prompt = teacher_forward_with_prompt(teacher_model, input_ids, attention_mask, prompt_module)

            teacher_outputs_no_prompt = teacher_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            )
            teacher_logits_no_prompt = teacher_outputs_no_prompt.logits

            student_outputs = student_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=None,
                use_cache=False,
            )
            student_logits = student_outputs.logits

            L_kd = distillation_loss(student_logits, teacher_logits_with_prompt)
            L_reg = regularization_loss(teacher_logits_with_prompt, teacher_logits_no_prompt)
            reg_weight = max(0.0, 1.0 - (current_step / total_steps))
            prompt_loss = L_kd + reg_weight * L_reg

            prompt_optimizer.zero_grad()
            scaler.scale(prompt_loss).backward(retain_graph=True)
            scaler.step(prompt_optimizer)
            scaler.update()

            L_student_kd = distillation_loss(student_logits, teacher_logits_with_prompt)
            L_student_supervised = supervised_loss(student_logits, labels)
            student_loss = 0.5 * L_student_kd + 0.5 * L_student_supervised

            student_optimizer.zero_grad()
            scaler.scale(student_loss).backward()
            scaler.step(student_optimizer)
            scaler.update()

        wandb.log({
            "prompt_loss": prompt_loss.item(),
            "student_loss": student_loss.item(),
            "step": current_step,
        })
        print(f"Step {current_step}: Prompt Loss {prompt_loss.item():.4f}, Student Loss {student_loss.item():.4f}")
        current_step += 1

        if current_step % VALIDATION_EVERY_N_STEPS == 0:
            student_model.eval()
            total_val_loss = 0.0
            val_steps = 0

            with torch.no_grad():
                for val_batch in val_loader:
                    val_input_ids = val_batch["input_ids"].to(device)
                    val_attention_mask = val_batch["attention_mask"].to(device)
                    val_labels = val_batch["labels"].to(device)

                    with autocast():
                        val_outputs = student_model(
                            input_ids=val_input_ids,
                            attention_mask=val_attention_mask,
                            labels=None,
                            use_cache=False,
                        )
                        val_logits = val_outputs.logits
                        val_loss = supervised_loss(val_logits, val_labels)

                    total_val_loss += val_loss.item()
                    val_steps += 1

            avg_val_loss = total_val_loss / val_steps
            print(f"Validation Loss at step {current_step}: {avg_val_loss:.4f}")
            wandb.log({"val_loss": avg_val_loss, "step": current_step})

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                print(f"Saving new best model at step {current_step} with val_loss {avg_val_loss:.4f}")

                if last_best_checkpoint_folder is not None:
                    import shutil
                    if os.path.exists(last_best_checkpoint_folder):
                        shutil.rmtree(last_best_checkpoint_folder)
                    if last_best_prompt_file is not None and os.path.exists(last_best_prompt_file):
                        os.remove(last_best_prompt_file)

                save_folder = f"/kaggle/working/best_student_checkpoint"
                save_prompt = f"/kaggle/working/best_prompt_checkpoint.pt"
                student_model.save_pretrained(save_folder)
                student_tokenizer.save_pretrained(save_folder)
                torch.save(prompt_module.state_dict(), save_prompt)
                last_best_checkpoint_folder = save_folder
                last_best_prompt_file = save_prompt
