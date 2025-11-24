import torch
from transformers import LlamaForCausalLM, LlamaTokenizer
from peft import PeftModel
import numpy as np

# -------------------------------
# 1️⃣ Load trained model
# -------------------------------
base_model_name = "decapoda-research/llama-7b-hf"
model_path = "./radar_genai_model"

tokenizer = LlamaTokenizer.from_pretrained(base_model_name)
model = LlamaForCausalLM.from_pretrained(base_model_name, device_map="auto", torch_dtype=torch.float16)
model = PeftModel.from_pretrained(model, model_path)
model.eval()

# -------------------------------
# 2️⃣ Example radar input
# -------------------------------
radar_frame = np.random.randint(0,2,(16,16)).flatten()
radar_str = " ".join(map(str, radar_frame.tolist()))
input_text = f"Radar data: {radar_str} -> Describe objects:"

inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

# -------------------------------
# 3️⃣ Generate description
# -------------------------------
with torch.no_grad():
    output_ids = model.generate(**inputs, max_new_tokens=64)
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

print("Generated Text:")
print(output_text)
