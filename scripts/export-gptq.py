import torch
import random
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

# Define base model and output directory
model_path  = "<model path>"
output_path = "<output path>"

# Load quantize config, model and tokenizer
quantization_config = BaseQuantizeConfig(
    bits=4,                 # quantize model to 4-bit
    group_size=128,         # it is recommended to set the value to 128
    desc_act=False,         # set to False can significantly speed up inference but the perplexity may slightly bad
    damp_percent=0.01,
)
model = AutoGPTQForCausalLM.from_pretrained(model_path, quantization_config)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# TODO Load data and tokenize examples
n_samples = 1024
data = None
tokenized_data = tokenizer(data, return_tensors='pt')

# Format tokenized examples
examples_ids = []
for _ in range(n_samples):
    i = random.randint(0, tokenized_data.input_ids.shape[1] - tokenizer.model_max_length - 1)
    j = i + tokenizer.model_max_length
    input_ids = tokenized_data.input_ids[:, i:j]
    attention_mask = torch.ones_like(input_ids)
    examples_ids.append({'input_ids': input_ids, 'attention_mask': attention_mask})

# Quantize with GPTQ
model.quantize(
    examples_ids,
    batch_size=1,
    use_triton=True,
)

# Save model and tokenizer
model.save_quantized(output_path, use_safetensors=True)
tokenizer.save_pretrained(output_path)