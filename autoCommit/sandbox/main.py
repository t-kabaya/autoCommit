print("start")

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from datasets import load_dataset
from pprint import pprint

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

def get_completion(query: str, model, tokenizer) -> str:
  device = "cuda:0"

  prompt_template = """
  <start_of_turn>user
  Below is an instruction that describes a task. Write a response that appropriately completes the request.
  {query}
  <end_of_turn>\n<start_of_turn>model


  """
  prompt = prompt_template.format(query=query)

  encodeds = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)

  model_inputs = encodeds.to(device)


  generated_ids = model.generate(**model_inputs, max_new_tokens=1000, do_sample=True, pad_token_id=tokenizer.eos_token_id)
  # decoded = tokenizer.batch_decode(generated_ids)
  decoded = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
  return (decoded)


result = get_completion(query="code the fibonacci series in python using reccursion", model=model, tokenizer=tokenizer)
print(result)

print("\nLoading dataset...")
dataset = load_dataset("TokenBender/code_instructions_122k_alpaca_style", split="train")
print(f"Dataset loaded. Total examples: {len(dataset)}")

print("\nFirst data example:")
pprint(dataset[0])
