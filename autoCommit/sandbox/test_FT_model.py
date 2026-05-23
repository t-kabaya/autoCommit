import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

model_id = "./models/gemma-2-2b"
tokenizer = AutoTokenizer.from_pretrained(model_id)
new_model = "gemma2-Code-Instruct-Finetune-test" #Name of the model you will be pushing to huggingface model hub

def get_completion(query: str, model, tokenizer) -> str:
    device = "cuda:0"
    prompt_template = """<start_of_turn>user
Below is an instruction that describes a task. Write a response that appropriately completes the request.
{query}
<end_of_turn>
<start_of_turn>model
"""
    prompt = prompt_template.format(query=query)
    encodeds = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
    model_inputs = encodeds.to(device)
    generated_ids = model.generate(**model_inputs, max_new_tokens=1000, do_sample=True, pad_token_id=tokenizer.eos_token_id)
    decoded = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return decoded

base_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    low_cpu_mem_usage=True,
    return_dict=True,
    torch_dtype=torch.float16,
    device_map={"": 0},
)

merged_model= PeftModel.from_pretrained(base_model, new_model)
merged_model= merged_model.merge_and_unload()

# Save the merged model
merged_model.save_pretrained("merged_model",safe_serialization=True)
tokenizer.save_pretrained("merged_model")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

result = get_completion(query="code the fibonacci series in python using reccursion", model=merged_model, tokenizer=tokenizer)

print(result)