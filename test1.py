from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_name = "Qwen/Qwen2.5-Coder-7B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    torch_dtype="auto", 
    device_map={"": "cpu"},  # Force model on CPU
    trust_remote_code=True
)
# Prepare a sample prompt
input_text = "def hello_world():\n    # This function should print 'Hello, world!'"

# Tokenize the input
inputs = tokenizer(input_text, return_tensors="pt").to(model.device)  # model.device should be CPU in this case

# Generate output (e.g., predict the next few lines of code)
# max_new_tokens defines how many tokens to generate, adjust as needed
outputs = model.generate(**inputs, max_new_tokens=50)

# Decode and print the generated text
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_text)
