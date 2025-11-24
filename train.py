import torch
from torch.utils.data import Dataset, DataLoader
from transformers import LlamaForCausalLM, LlamaTokenizer, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType
import numpy as np

# -------------------------------
# 1️⃣ Dataset
# -------------------------------
class RadarTextDataset(Dataset):
    """
    Simple radar-to-text dataset
    """
    def __init__(self, num_samples=50):
        self.num_samples = num_samples
        self.data = []
        for i in range(num_samples):
            radar_frame = np.random.randint(0, 2, (16,16)).flatten()
            description = f"There are {radar_frame.sum()} objects in the scene."
            self.data.append((radar_frame, description))

        self.tokenizer = LlamaTokenizer.from_pretrained("decapoda-research/llama-7b-hf")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        radar_frame, description = self.data[idx]
        radar_str = " ".join(map(str, radar_frame.tolist()))
        input_text = f"Radar data: {radar_str} -> Describe objects:"
        inputs = self.tokenizer(input_text, truncation=True, padding="max_length", max_length=256, return_tensors="pt")
        labels = self.tokenizer(description, truncation=True, padding="max_length", max_length=64, return_tensors="pt")
        inputs['labels'] = labels['input_ids']
        return {k: v.squeeze(0) for k, v in inputs.items()}

dataset = RadarTextDataset()
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# -------------------------------
# 2️⃣ Load Pretrained LLM
# -------------------------------
model_name = "decapoda-research/llama-7b-hf"
model = LlamaForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.float16
)

# -------------------------------
# 3️⃣ LoRA adaptation
# -------------------------------
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
