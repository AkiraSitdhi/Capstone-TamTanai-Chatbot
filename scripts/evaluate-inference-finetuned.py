import os
import time
import torch
import pandas as pd
import bitsandbytes as bnb

from tqdm import tqdm
from torch.nn import DataParallel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"using {DEVICE} device")

model_name_or_path = "../tamtanai"

# Load both LLM model and tokenizer
def load_LLM_and_tokenizer():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        quantization_config=bnb_config,
        local_files_only=True,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        use_cache=False
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        padding_side="left",
        add_eos_token=True,
        add_bos_token=True,
    )
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

model, tokenizer = load_LLM_and_tokenizer()
model.config.use_cache = False

if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    model = DataParallel(model).module


def generate_answer_with_timer(text: str):
    try:
        start_time = time.time()
        encoded_input = tokenizer.encode(text, return_tensors='pt', add_special_tokens=False).to(DEVICE)
        with torch.cuda.amp.autocast():
            generate_kwargs = dict(
                {"input_ids": encoded_input},
                do_sample=True,
                max_new_tokens=512,
                top_p=0.9,
                temperature=0.7,
                repetition_penalty=1.1,
                use_cache=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        encoded_output = model.generate(**generate_kwargs)
        response = tokenizer.decode(encoded_output[0][len(encoded_input[0]):], skip_special_tokens=True)
        response_time = time.time() - start_time
        return response, response_time
    except:
        print("Encounter error!")
        return None, 0.0


test_data = pd.read_csv("../testdata.csv")
print("Finished reading data")

answers = []
times = []
for i, (index, data_point) in tqdm(enumerate(test_data.iterrows())):
    prompt = data_point["prompt"]
    output = generate_answer_with_timer(prompt)
    answers.append(output[0])
    times.append(output[1])

test_data["response_finetune"] = answers
test_data["time_finetune"] = times

test_data.to_csv("./evaluate_inference_finetuned_seallmv2.csv", index=False, encoding='utf-8')
print("Finished inference test dataset")