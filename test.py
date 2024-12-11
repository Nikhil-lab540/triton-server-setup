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

# Prepare dummy input for ONNX export (CPU)
dummy_input = tokenizer("def hello_world():", return_tensors="pt").input_ids

# Export to ONNX (CPU)
onnx_path = "./qwen2.5_coder.onnx"
torch.onnx.export(
    model, 
    (dummy_input,), 
    onnx_path,
    input_names=["input_ids"],
    output_names=["output"],
    dynamic_axes={
        "input_ids": {0: "batch_size", 1: "seq_length"},
        "output": {0: "batch_size", 1: "seq_length"}
    },
    opset_version=13
)

print("Model exported to ONNX format:", onnx_path)
