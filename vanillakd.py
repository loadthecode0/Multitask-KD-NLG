import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader
from datasets import load_dataset
from tqdm import tqdm
import wandb
from peft import PeftModel, PeftConfig, get_peft_model, LoraConfig, TaskType
from torch.cuda.amp import autocast, GradScaler
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 4

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
    target_modules=["q_proj", "v_proj"],  # adjust based on model architecture
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)


student_model = get_peft_model(student_base_model, student_lora_config).to(device)
student_tokenizer = AutoTokenizer.from_pretrained(student_model_path)
student_tokenizer.pad_token = student_tokenizer.eos_token


def distillation_loss(student_logits, teacher_logits, labels=None, temperature=2.0):
    """
    Improved knowledge distillation loss with proper temperature scaling and masking
    
    Args:
        student_logits: logits from student model [batch_size, seq_len, vocab_size]
        teacher_logits: logits from teacher model [batch_size, seq_len, vocab_size]
        labels: labels tensor for masking (-100 indicates tokens to ignore)
        temperature: softmax temperature for distillation
    """
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
    

    flat_logits = shift_logits.reshape(-1, shift_logits.shape[-1])  # [batch_size*seq_len-1, vocab_size]
    flat_labels = shift_labels.reshape(-1)  # [batch_size*seq_len-1]

    losses = SUP_LOSS_FUNC(flat_logits, flat_labels)  # [batch_size*seq_len-1]
    

    token_losses = losses.view(shift_labels.shape)  # [batch_size, seq_len-1]
    mask = (shift_labels != -100).float()  # [batch_size, seq_len-1]

    masked_loss = (token_losses * mask).sum() / (mask.sum() + 1e-6)
    return masked_loss


dataset = load_dataset("json", data_files="instruction_combined_1.jsonl")['train']
EOS_TOKEN = student_tokenizer.eos_token
dataset = dataset.shuffle(seed=120).select(range(int(0.03 * len(dataset))))
split_dataset = dataset.train_test_split(test_size=0.1, seed=121)
train_data, val_data = split_dataset["train"], split_dataset["test"]

def preprocess_function(example, tokenizer):
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
    
    # Create labels by masking the prompt part with -100
    labels = input_ids.clone()
    prompt_length = len(tokenizer(prompt)["input_ids"])
    labels[:, :prompt_length] = -100

    return {
        "input_ids": input_ids.squeeze(0),
        "attention_mask": inputs_enc["attention_mask"].squeeze(0),
        "labels": labels.squeeze(0),
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
        }
    )

train_loader_teacher = create_dataloader(train_dataset_teacher, BATCH_SIZE)
val_loader_teacher = create_dataloader(val_dataset_teacher, BATCH_SIZE, shuffle=False)
train_loader_student = create_dataloader(train_dataset_student, BATCH_SIZE)
val_loader_student = create_dataloader(val_dataset_student, BATCH_SIZE, shuffle=False)

# Optimizer
student_optimizer = torch.optim.AdamW(student_model.parameters(), lr=5e-5)

total_steps = 10000
current_step = 0
VALIDATION_EVERY_N_STEPS = 200
best_val_loss = float('inf')
last_best_checkpoint_folder = None


wandb.init(
    project="promptkd-distillation",
    name="improved-vanilla-kd",
    config={
        "epochs": 2,
        "batch_size": BATCH_SIZE,
        "learning_rate": 5e-5,
        "student_model": student_model_path,
        "teacher_model": teacher_model_path,
        "distillation_temperature": 2.0,
        "alpha_weight": 0.5,  # Weight for balancing supervised and distillation loss
    }
)


steps_per_epoch = min(len(train_loader_student), len(train_loader_teacher))
for epoch in range(2):
    for (batch_student, batch_teacher) in tqdm(zip(train_loader_student, train_loader_teacher), 
                                              total=steps_per_epoch, 
                                              desc=f"Epoch {epoch+1}"):
        student_model.train()
        

        input_ids_student = batch_student["input_ids"].to(device)
        input_ids_teacher = batch_teacher["input_ids"].to(device)
        attention_mask_student = batch_student["attention_mask"].to(device)
        attention_mask_teacher = batch_teacher["attention_mask"].to(device)
        labels = batch_student["labels"].to(device)
        
        with autocast():
            # Get teacher output logits (with gradient disabled)
            with torch.no_grad():
                teacher_outputs = teacher_model(
                    input_ids=input_ids_teacher,
                    attention_mask=attention_mask_teacher,
                    use_cache=False,
                )
                teacher_logits = teacher_outputs.logits
            
            # Get student output logits
            student_outputs = student_model(
                input_ids=input_ids_student,
                attention_mask=attention_mask_student,
                labels=None,
                use_cache=False,
            )
            student_logits = student_outputs.logits
            
            # Calculate losses
            L_student_supervised = supervised_loss(student_logits, labels)
            L_student_kd = distillation_loss(
                student_logits, 
                teacher_logits,
                labels=labels,
                temperature=2.0
            )
            
            # Balance between supervised and distillation loss
            alpha = 0.5  # Adjust this weight as needed
            student_loss = (1 - alpha) * L_student_supervised + alpha * L_student_kd
        
        # Backward and optimize
        student_optimizer.zero_grad()
        student_loss.backward()
        student_optimizer.step()
        
        # Log metrics
        wandb.log({
            "distill_loss": L_student_kd.item(),
            "sup_loss": L_student_supervised.item(),
            "total_loss": student_loss.item(),
            "step": current_step,
        })
        
        # Print progress
        if (current_step % 25 == 0):
            print(f"Step {current_step}: KD Loss {L_student_kd.item():.4f}, "
                  f"Supervised Loss {L_student_supervised.item():.4f}, "
                  f"Total Loss {student_loss.item():.4f}")
        
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
            #     best_val_loss = avg_val_loss
            #     print(f"Saving new best model at step {current_step} with val_loss {avg_val_loss:.4f}")
                
                # Clean up previous best checkpoint
                # if last_best_checkpoint_folder is not None:
                #     import shutil
                #     if os.path.exists(last_best_checkpoint_folder):
                #         shutil.rmtree(last_best_checkpoint_folder)
                
            # Save new best checkpoint
            save_folder = f"vanilla/vanilla_student_checkpoint_{current_step}"
            student_model.save_pretrained(save_folder)
            student_tokenizer.save_pretrained(save_folder)
            last_best_checkpoint_folder = save_folder

print("Training complete!")
