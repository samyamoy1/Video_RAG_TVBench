import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"

print("Loading Qwen 2.5 VL 3B...")
print("GPU:", torch.cuda.get_device_name(0))

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    device_map="auto",
)

processor = AutoProcessor.from_pretrained(MODEL_ID)

print("\nQwen model loaded successfully!")
print("Model device:", next(model.parameters()).device)

allocated = torch.cuda.memory_allocated() / 1024**3
reserved = torch.cuda.memory_reserved() / 1024**3

print(f"GPU memory allocated: {allocated:.2f} GB")
print(f"GPU memory reserved: {reserved:.2f} GB")