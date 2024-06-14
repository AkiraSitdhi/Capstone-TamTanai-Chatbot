import os
import time
import torch
import pandas as pd
from torch.nn import DataParallel
from sklearn.model_selection import train_test_split
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

print('import completed')

# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"using {DEVICE} device")

model_path = "<model path>"
finetune_weight_path = "<finetune path>"

tokenizer = AutoTokenizer.from_pretrained(finetune_weight_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    local_files_only=True,
    load_in_4bit = True,
    device_map = 'cuda:0',
    use_cache = False,
    torch_dtype = torch.bfloat16
)
model = PeftModel.from_pretrained(model, finetune_weight_path)

print('load model completed')

# if torch.cuda.device_count() > 1:
#     model = DataParallel(model).module

law_dataset = pd.read_csv("<data path>", encoding='utf-8')
train_ratio = 0.8
validation_ratio = 0.1
test_ratio = 0.1
train_data, temp_data = train_test_split(law_dataset, test_size=1 - train_ratio, random_state=42)
validation_data, test_data = train_test_split(temp_data, test_size=test_ratio/(test_ratio + validation_ratio), random_state=42)

torch.cuda.empty_cache()

INFERENCE_SYSTEM_PROMPT = """คุณคือนักกฎหมายที่จะตอบคำถามเกี่ยวกับกฎหมาย จงตอบคำถามโดยใช้ความรู้ที่ให้ดังต่อไปนี้
ถ้าหากคุณไม่รู้คำตอบ ให้ตอบว่าไม่รู้ อย่าสร้างคำตอบขึ้นมาเอง"""


def generate_inference_prompt(
    question: str,
    knowledge: str,
    system_prompt: str = INFERENCE_SYSTEM_PROMPT
) -> str:
    return f"""<s><|im_start|>system
{system_prompt.strip()}\nความรู้ที่ให้:
{knowledge.strip()}</s><|im_start|>user
{question.strip()}</s><|im_start|>assistant
"""

def inference(text: str):
    try:
        encoded_input = tokenizer.encode(text, return_tensors='pt', add_special_tokens=False).to(DEVICE)
        start_time = time.time()
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
        response_time = time.time()
        ans = tokenizer.decode(encoded_output[0][len(encoded_input[0]):])

        del encoded_input
        del encoded_output 
        torch.cuda.empty_cache()

        return ans, response_time - start_time
    except:
        return None, 0.0
    
print('begin inference')

answers = []
times = []
for i, (index, data_point) in tqdm(enumerate(test_data.iterrows())):
    question = data_point["question"]
    label = data_point["answer"]
    references = data_point["references"]
    source = data_point["source"]
    knowledge = data_point["knowledges"]
    ans, tt = inference(generate_inference_prompt(question, knowledge))
    answers.append(ans)
    times.append(tt)

test_data["response_finetune"] = answers
test_data["time_finetune"] = times
test_data.to_csv("<result export path>", index=False, encoding='utf-8')

print('\ninference completed')