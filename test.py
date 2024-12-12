#!/usr/bin/env python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import onnx
import onnxruntime as ort
import numpy as np
torch.set_num_threads(8)
# -----------------------------
# Step 1: Specify the model
# -----------------------------
model_name = "Qwen/Qwen2.5-Coder-7B"
device =  "cpu"
# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,  # FP16 precision
    device_map={"": device},
   # Load on CPU to reduce VRAM consumption
    trust_remote_code=True
)
model.eval()

# -----------------------------
# Step 2: Prepare dummy input
# -----------------------------
input_text = "def hello_world():"
inputs = tokenizer(input_text, return_tensors="pt")

# Prepare dummy inputs for export
dummy_input_ids = inputs["input_ids"].to(device)

# -----------------------------
# Step 3: Export the model to ONNX with external data
# -----------------------------
onnx_path = "qwen2.5_coder.onnx"
torch.onnx.export(
    model,
    (dummy_input_ids,),
    "qwen2.5_coder.onnx",   # Path for main ONNX file
    input_names=["input_ids"],
    output_names=["output"],
    dynamic_axes={
        "input_ids": {0: "batch_size", 1: "seq_length"},
        "output": {0: "batch_size", 1: "seq_length"}
    },
    opset_version=14,
    export_params=True,      # Export parameters
    do_constant_folding=True,
    large_model=True,        # Required for large models
    save_external_data=True, # Explicitly save external data
    all_tensors_to_one_file=False,  # Split external data into multiple files
    output_file_path="qwen2.5_coder.onnx"  # Explicit file path
)



print(f"Model successfully exported to ONNX format: {onnx_path}")

# -----------------------------
# Step 4: Validate the ONNX model
# -----------------------------
model_onnx = onnx.load(onnx_path)
onnx.checker.check_model(model_onnx)
print("ONNX model validation successful.")

# -----------------------------
# Step 5: Test with ONNX Runtime
# -----------------------------
input_ids_np = dummy_input_ids.cpu().numpy()

# Create an ONNX Runtime session
ort_session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

# Run inference
onnx_outputs = ort_session.run(None, {"input_ids": input_ids_np})
print("ONNX inference output shape:", onnx_outputs[0].shape)
