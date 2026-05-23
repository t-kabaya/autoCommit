
new_model = "gemma2-Code-Instruct-Finetune-test" #Name of the model you will be pushing to huggingface model hub

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