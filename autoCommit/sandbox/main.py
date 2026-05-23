print("start")

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device count: {torch.cuda.device_count()}")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model_id = "./models/gemma-2-2b"  # ローカルモデルパスに変更

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_id, add_eos_token=True)
print("Tokenizer loaded")

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    low_cpu_mem_usage=True
)
print("Model loaded successfully")