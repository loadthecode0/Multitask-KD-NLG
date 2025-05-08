import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
from peft import PeftModel, PeftConfig

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the fine-tuned model and tokenizer
model_path = "./outputs/checkpoint-500"

# Load the PEFT config to find the base model
peft_model_path = model_path  # e.g., "./checkpoints/lora-student"
config = PeftConfig.from_pretrained(peft_model_path)

# Load the base model that LoRA was trained on
base_model = AutoModelForCausalLM.from_pretrained(
    config.base_model_name_or_path,
    torch_dtype=torch.float16
).to(device)

# Apply the LoRA adapters
model = PeftModel.from_pretrained(base_model, peft_model_path).to(device)
# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)



# model = AutoModelForCausalLM.from_pretrained(
#     model_path,
#     torch_dtype=torch.float16,  # or bfloat16 depending on your GPU
#     device_map="auto"           # lets Transformers automatically place layers on available GPUs
# )
# tokenizer = AutoTokenizer.from_pretrained(model_path)

# Enable evaluation/inference mode
model.eval()

# Get EOS token (important for proper formatting)
EOS_TOKEN = tokenizer.eos_token

# Prompt formatting function
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
"""
    else:
        prompt = f"""### Instruction:
{instruction}

### Response:
"""
    return {"text": prompt}

# Example input for inference
inf_examples = [

{
    "instruction": "Summarize the following news article.",
    "input": "(CNN)After security breaches involving firearms, the FAA has suspended a program that allows safety inspectors to bypass the Transportation Security Administration's screening checkpoints, the agency announced Friday. A Federal Aviation Administration inspector was arrested January 13 after TSA employees at LaGuardia Airport in New York discovered a firearm in his carry-on bag, the TSA said. The inspector had used his Security Identification Display Area badge to skip TSA security checkpoints at Atlanta Hartsfield-Jackson Airport, the TSA said. He didn't have SIDA clearance for LaGuardia, the TSA said. Two men were arrested last month after smuggling more than 100 firearms from the Atlanta airport to New York, authorities said. One of the men was an airport employee who skipped security checkpoints as part of the smuggling operation, authorities said.",
    "output": "FAA says safety inspectors will not be allowed to bypass TSA security checkpoints.\nAn FAA inspector was arrested January 13 with a firearm in his carry-on bag, TSA says."
},

    {"instruction": "Answer the question using the context. If the answer is not present, say 'unanswerable'.", "input": "Question: To set the record for Grammys, how many did Beyonce win?\nContext: Following the disbandment of Destiny's Child in June 2005, she released her second solo album, B'Day (2006), which contained hits \"D\u00e9j\u00e0 Vu\", \"Irreplaceable\", and \"Beautiful Liar\". Beyonc\u00e9 also ventured into acting, with a Golden Globe-nominated performance in Dreamgirls (2006), and starring roles in The Pink Panther (2006) and Obsessed (2009). Her marriage to rapper Jay Z and portrayal of Etta James in Cadillac Records (2008) influenced her third album, I Am... Sasha Fierce (2008), which saw the birth of her alter-ego Sasha Fierce and earned a record-setting six Grammy Awards in 2010, including Song of the Year for \"Single Ladies (Put a Ring on It)\". Beyonc\u00e9 took a hiatus from music in 2010 and took over management of her career; her fourth album 4 (2011) was subsequently mellower in tone, exploring 1970s funk, 1980s pop, and 1990s soul. Her critically acclaimed fifth studio album, Beyonc\u00e9 (2013), was distinguished from previous releases by its experimental production and exploration of darker themes.", "output": "six"},

    {"instruction": "Answer the question using the context. If the answer is not present, say 'unanswerable'.", "input": "Question: What song won Best R&B Performance in the 43 Annual Grammy Awards?\nContext: The group changed their name to Destiny's Child in 1996, based upon a passage in the Book of Isaiah. In 1997, Destiny's Child released their major label debut song \"Killing Time\" on the soundtrack to the 1997 film, Men in Black. The following year, the group released their self-titled debut album, scoring their first major hit \"No, No, No\". The album established the group as a viable act in the music industry, with moderate sales and winning the group three Soul Train Lady of Soul Awards for Best R&B/Soul Album of the Year, Best R&B/Soul or Rap New Artist, and Best R&B/Soul Single for \"No, No, No\". The group released their multi-platinum second album The Writing's on the Wall in 1999. The record features some of the group's most widely known songs such as \"Bills, Bills, Bills\", the group's first number-one single, \"Jumpin' Jumpin'\" and \"Say My Name\", which became their most successful song at the time, and would remain one of their signature songs. \"Say My Name\" won the Best R&B Performance by a Duo or Group with Vocals and the Best R&B Song at the 43rd Annual Grammy Awards. The Writing's on the Wall sold more than eight million copies worldwide. During this time, Beyonc\u00e9 recorded a duet with Marc Nelson, an original member of Boyz II Men, on the song \"After All Is Said and Done\" for the soundtrack to the 1999 film, The Best Man.", "output": "Say My Name"},


{"instruction": "Answer the question using the context. If the answer is not present, say 'unanswerable'.", "input": "Question: What did she agree to do for 50 million dollars in 2012?\nContext: Beyonc\u00e9 has worked with Pepsi since 2002, and in 2004 appeared in a Gladiator-themed commercial with Britney Spears, Pink, and Enrique Iglesias. In 2012, Beyonc\u00e9 signed a $50 million deal to endorse Pepsi. The Center for Science in the Public Interest (CSPINET) wrote Beyonc\u00e9 an open letter asking her to reconsider the deal because of the unhealthiness of the product and to donate the proceeds to a medical organisation. Nevertheless, NetBase found that Beyonc\u00e9's campaign was the most talked about endorsement in April 2013, with a 70 per cent positive audience response to the commercial and print ads.", "output": "endorse Pepsi"},

    {"instruction": "Rewrite the sentence to mean the same thing in different words.", "input": "What is the Sahara, and how do the average temperatures there compare to the ones in the Antarctica?", "output": "What is the Sahara, and how do the average temperatures there compare to the ones in the Simpson Desert?"},

    {"instruction": "Rewrite the sentence to mean the same thing in different words.", "input": "Why is it important to get good grades?", "output": "Why is it good to get good grades?"},

    

{"instruction": "Rewrite the sentence to mean the same thing in different words.", "input": "Have you ever taken revenge? How and why did you take it?", "output": "Have you ever taken revenge? How?"},



]

for inf_example in inf_examples:
    # Tokenize input prompt
    formatted_prompt = formatting_prompts_func(inf_example)["text"]
    inputs = tokenizer([formatted_prompt], return_tensors="pt").to(model.device)
    
    # Text generation with streaming output
    streamer = TextStreamer(tokenizer)
    _ = model.generate(**inputs, streamer=streamer, max_new_tokens=128)
