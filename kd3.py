import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import wandb
from peft import PeftModel, PeftConfig, get_peft_model, LoraConfig, TaskType
from tqdm import tqdm

BATCH_SIZE = 2
NUM_TASKS = 3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Running on device:", device)

# CONFIG =================
teacher_model_path = "./teacher-8B"
student_model_path = "/home/models/Llama-3.2-1B"
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


student_base_model = AutoModelForCausalLM.from_pretrained(
    student_model_path,
    torch_dtype=torch.float16
).to(device)


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


class PromptEmbedding(nn.Module):
    def __init__(self, prompt_length, hidden_size):
        super().__init__()
        print(f'Initializing prompt embeddings with dimensions: {prompt_length}, {hidden_size}')
        self.prompt_embeddings = nn.Parameter(torch.randn(NUM_TASKS, prompt_length, hidden_size) * 0.02)
    
    def forward(self, task_ids):
        return self.prompt_embeddings[task_ids]

prompt_length = 20
hidden_size = teacher_model.get_input_embeddings().embedding_dim
prompt_module = PromptEmbedding(prompt_length, hidden_size).to(device)


def distillation_loss(student_logits, teacher_logits, labels=None, temperature=2.0):

    shift_student = student_logits[..., :-1, :].float()  # [batch_size, seq_len-1, vocab_size]
    shift_teacher = teacher_logits[..., :-1, :].float()  # [batch_size, seq_len-1, vocab_size]
    shift_labels = labels[..., 1:] if labels is not None else None  # [batch_size, seq_len-1]

    scaled_student = shift_student / temperature
    scaled_teacher = shift_teacher / temperature
    

    log_probs_student = F.log_softmax(scaled_student, dim=-1)  # [batch_size, seq_len-1, vocab_size]
    probs_teacher = F.softmax(scaled_teacher, dim=-1)  # [batch_size, seq_len-1, vocab_size]
    

    kl_div = F.kl_div(
        log_probs_student.view(-1, log_probs_student.size(-1)),
        probs_teacher.view(-1, probs_teacher.size(-1)),
        reduction="none"
    ).sum(dim=-1)  # [batch_size * seq_len-1]
    
    # Reshape back
    kl_div = kl_div.view(shift_student.size(0), -1)  # [batch_size, seq_len-1]
    

    if shift_labels is not None:
        mask = (shift_labels != -100).float()
    else:
        mask = torch.ones_like(kl_div)
    masked_kl_div = (kl_div * mask).sum() / (mask.sum() + 1e-6)
    return masked_kl_div * (temperature ** 2)

SUP_LOSS_FUNC = nn.CrossEntropyLoss(ignore_index=-100, reduction="none")
def supervised_loss(student_logits, labels):
    """
    Standard supervised cross-entropy loss with proper reshaping
    """
    shift_logits = student_logits[..., :-1, :].float()  # [batch_size, seq_len-1, vocab_size]
    shift_labels = labels[..., 1:]  # [batch_size, seq_len-1]
    
    # Reshape for loss calculation
    flat_logits = shift_logits.reshape(-1, shift_logits.shape[-1])  # [batch_size*seq_len-1, vocab_size]
    flat_labels = shift_labels.reshape(-1)  # [batch_size*seq_len-1]
    
    losses = SUP_LOSS_FUNC(flat_logits, flat_labels)  # [batch_size*seq_len-1]
    

    token_losses = losses.view(shift_labels.shape)  # [batch_size, seq_len-1]
    mask = (shift_labels != -100).float()  # [batch_size, seq_len-1]
    

    masked_loss = (token_losses * mask).sum() / (mask.sum() + 1e-6)
    return masked_loss

def regularization_loss(teacher_logits_with_prompt, teacher_logits_no_prompt, labels=None, temperature=1.0):

    student_logits = teacher_logits_with_prompt
    teacher_logits = teacher_logits_no_prompt

    shift_student = student_logits[..., :-1, :].float()  # [batch_size, seq_len-1, vocab_size]
    shift_teacher = teacher_logits[..., :-1, :].float()  # [batch_size, seq_len-1, vocab_size]
    shift_labels = labels[..., 1:] if labels is not None else None  # [batch_size, seq_len-1]
    
    # Apply temperature scaling
    scaled_student = shift_student / temperature
    scaled_teacher = shift_teacher / temperature
    
    log_probs_student = F.log_softmax(scaled_student, dim=-1)  # [batch_size, seq_len-1, vocab_size]
    probs_teacher = F.softmax(scaled_teacher, dim=-1)  # [batch_size, seq_len-1, vocab_size]

    kl_div = F.kl_div(
        log_probs_student.view(-1, log_probs_student.size(-1)),
        probs_teacher.view(-1, probs_teacher.size(-1)),
        reduction="none"
    ).sum(dim=-1)  # [batch_size * seq_len-1]
    

    kl_div = kl_div.view(shift_student.size(0), -1)  # [batch_size, seq_len-1]

    if shift_labels is not None:
        mask = (shift_labels != -100).float()
    else:
        mask = torch.ones_like(kl_div)
    
    masked_kl_div = (kl_div * mask).sum() / (mask.sum() + 1e-6)
    return masked_kl_div * (temperature ** 2)

def teacher_forward_with_prompt(teacher_model, input_ids, task_ids, attention_mask, prompt_module):
    """
    Forward pass through teacher model with learned prompt prepended to input
    """
    batch_size = input_ids.size(0)

    inputs_embeds = teacher_model.get_input_embeddings()(input_ids)

    prompt_embeds = prompt_module(task_ids)

    inputs_embeds = torch.cat([prompt_embeds, inputs_embeds], dim=1)
    

    prompt_attention = torch.ones(batch_size, prompt_embeds.size(1)).to(attention_mask.device)
    attention_mask = torch.cat([prompt_attention, attention_mask], dim=1)


    outputs = teacher_model(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        use_cache=False,
    )
    
    return outputs.logits


dataset = load_dataset("json", data_files="instruction_combined_1.jsonl")['train']
EOS_TOKEN = student_tokenizer.eos_token
dataset = dataset.shuffle(seed=120).select(range(int(0.03 * len(dataset))))
split_dataset = dataset.train_test_split(test_size=0.1, seed=121)
train_data, val_data = split_dataset["train"], split_dataset["test"]

def get_task_id(instruction: str) -> int:
    """
    Determine the task ID based on instruction keywords
    """
    instruction = instruction.lower()
    if "summarize" in instruction or "summary" in instruction:
        return 0
    elif "rewrite" in instruction:
        return 1
    elif "question" in instruction or "answer" in instruction:
        return 2
    else:
        return 0  # Default to task 0 instead of -1 to avoid index errors

def preprocess_function(example, tokenizer):
    """
    Unified preprocessing function for both teacher and student
    """

    prompt = f"""### Instruction:
{example["instruction"]}

### Input:
{example["input"]}

### Response: 
"""
    response = example["output"]
    full_text = prompt + response + EOS_TOKEN

    inputs_enc = tokenizer(full_text, truncation=True, padding="max_length", max_length=512, return_tensors="pt")
    input_ids = inputs_enc["input_ids"]
    
    labels = input_ids.clone()
    prompt_length = len(tokenizer(prompt)["input_ids"])
    labels[:, :prompt_length] = -100

    return {
        "input_ids": input_ids.squeeze(0),
        "attention_mask": inputs_enc["attention_mask"].squeeze(0),
        "labels": labels.squeeze(0),
        "task_type": get_task_id(example["instruction"]),
    }

train_dataset_teacher = train_data.map(lambda x: preprocess_function(x, teacher_tokenizer))
val_dataset_teacher = val_data.map(lambda x: preprocess_function(x, teacher_tokenizer)) 
train_dataset_student = train_data.map(lambda x: preprocess_function(x, student_tokenizer))
val_dataset_student = val_data.map(lambda x: preprocess_function(x, student_tokenizer))


for dataset in [train_dataset_teacher, val_dataset_teacher, train_dataset_student, val_dataset_student]:
    dataset.set_format(type="torch")


def create_dataloader(dataset, batch_size, shuffle=True):
    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        collate_fn=lambda x: {
            "input_ids": torch.stack([item["input_ids"] for item in x]),
            "attention_mask": torch.stack([item["attention_mask"] for item in x]),
            "labels": torch.stack([item["labels"] for item in x]),
            "task_id": torch.tensor([item["task_type"] for item in x], dtype=torch.long),
        }
    )

train_loader_teacher = create_dataloader(train_dataset_teacher, BATCH_SIZE)
val_loader_teacher = create_dataloader(val_dataset_teacher, BATCH_SIZE, shuffle=False)
train_loader_student = create_dataloader(train_dataset_student, BATCH_SIZE)
val_loader_student = create_dataloader(val_dataset_student, BATCH_SIZE, shuffle=False)

prompt_optimizer = torch.optim.AdamW(prompt_module.parameters(), lr=1e-5)  # Changed from 5e-5
student_optimizer = torch.optim.AdamW(student_model.parameters(), lr=1e-5)  # Changed from 5e-5

total_steps = 10000
current_step = 0
VALIDATION_EVERY_N_STEPS = 200
scaler = GradScaler()
best_val_loss = float('inf')
last_best_checkpoint_folder = None
last_best_prompt_file = None

wandb.init(
    project="promptkd-distillation",
    name="improved-promptkd",
    config={
        "epochs": 2,
        "batch_size": BATCH_SIZE,
        "learning_rate": 1e-5,  # Updated learning rate
        "prompt_length": prompt_length,
        "student_model": student_model_path,
        "teacher_model": teacher_model_path,
        "distillation_temperature": 1.0,  # Updated temperature
        "alpha_weight": 0.5,  # Weight for balancing supervised and distillation loss
    }
)

steps_per_epoch = min(len(train_loader_student), len(train_loader_teacher))
for epoch in range(2):  # Reduced to 2 epochs to match vanilla KD
    for (batch_student, batch_teacher) in tqdm(zip(train_loader_student, train_loader_teacher), 
                                              total=steps_per_epoch, 
                                              desc=f"Epoch {epoch+1}"):
        input_ids_student = batch_student["input_ids"].to(device)
        input_ids_teacher = batch_teacher["input_ids"].to(device)
        task_ids = batch_teacher["task_id"].to(device)
        attention_mask_student = batch_student["attention_mask"].to(device)
        attention_mask_teacher = batch_teacher["attention_mask"].to(device)
        labels = batch_student["labels"].to(device)
        
        with autocast():
            teacher_logits_with_prompt = teacher_forward_with_prompt(
                teacher_model, input_ids_teacher, task_ids, attention_mask_teacher, prompt_module
            )
            teacher_logits_with_prompt = teacher_logits_with_prompt[:, prompt_length:, :]
            
            teacher_outputs_no_prompt = teacher_model(
                input_ids=input_ids_teacher,
                attention_mask=attention_mask_teacher,
                use_cache=False,
            )
            teacher_logits_no_prompt = teacher_outputs_no_prompt.logits
            
            # 3. Student forward pass
            student_outputs = student_model(
                input_ids=input_ids_student,
                attention_mask=attention_mask_student,
                labels=None,
                use_cache=False,
            )
            student_logits = student_outputs.logits

            L_kd = distillation_loss(student_logits, teacher_logits_with_prompt, labels)
            L_reg = regularization_loss(teacher_logits_with_prompt, teacher_logits_no_prompt, labels)
            
            reg_weight = max(0.0, 1.0 - (current_step / total_steps))
            prompt_loss = L_kd + reg_weight * L_reg
            
            L_student_kd = distillation_loss(student_logits, teacher_logits_with_prompt)
            L_student_supervised = supervised_loss(student_logits, labels)
            
            alpha = min(0.8, current_step / (total_steps * 0.5))  # Gradually increase to 0.8
            student_loss = alpha * L_student_kd + (1 - alpha) * L_student_supervised
        

        prompt_optimizer.zero_grad()
        scaler.scale(prompt_loss).backward(retain_graph=True)
        scaler.step(prompt_optimizer)
        
        student_optimizer.zero_grad()
        scaler.scale(student_loss).backward()
        scaler.step(student_optimizer)
        
        scaler.update()
        
        # Log metrics
        wandb.log({
            "prompt_loss": prompt_loss.item(),
            "distill_loss": L_student_kd.item(),
            "sup_loss": L_student_supervised.item(),
            "student_loss": student_loss.item(),
            "regularization_weight": reg_weight,
            "alpha_kd_weight": alpha,  # Log the changing alpha value
            "step": current_step,
        })
        
        # Print progress
        if (current_step % 25 == 0):
            print(f"Step {current_step}: Prompt Loss {prompt_loss.item():.4f}, "
                  f"KD Loss {L_student_kd.item():.4f}, "
                  f"Supervised Loss {L_student_supervised.item():.4f}, "
                  f"Total Student Loss {student_loss.item():.4f}, "
                  f"Alpha: {alpha:.2f}")
        
        current_step += 1
        
        # Validation
        if current_step % VALIDATION_EVERY_N_STEPS == 0:
            student_model.eval()
            total_val_loss = 0.0
            val_steps = 0
            
            with torch.no_grad():
                for val_batch in val_loader_student:
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
            
            # Save best model
            # if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            print(f"Saving new best model at step {current_step} with val_loss {avg_val_loss:.8f}")
            
            # # Clean up previous best checkpoint
            # if last_best_checkpoint_folder is not None:
            #     import shutil
            #     if os.path.exists(last_best_checkpoint_folder):
            #         shutil.rmtree(last_best_checkpoint_folder)
            #     if last_best_prompt_file is not None and os.path.exists(last_best_prompt_file):
            #         os.remove(last_best_prompt_file)
            
            # Save new best checkpoint
            save_folder = f"kd3/improved_promptkd_best_student_checkpoint_{current_step}"
            save_prompt = f"kd3/improved_promptkd_best_prompt_checkpoint_{current_step}.pt"
            student_model.save_pretrained(save_folder)
            student_tokenizer.save_pretrained(save_folder)
            torch.save(prompt_module.state_dict(), save_prompt)
            last_best_checkpoint_folder = save_folder
            last_best_prompt_file = save_prompt

print("Training complete!")
