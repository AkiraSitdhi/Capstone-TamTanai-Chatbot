import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print('import completed\n')

# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

model_path = "<model path>"
onnx_export_path = "<onnx export path>"

tokenizer = AutoTokenizer.from_pretrained(model_path)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float32,
)

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    # quantization_config=bnb_config,
    local_files_only=True,
    device_map = 'auto',
    pad_token_id=tokenizer.eos_token_id,
    torch_dtype=torch.float32,
)

print('load model (combine weight) completed\n')

# test inference
print("Test inference\n")

text = """<s><|im_start|>system
คุณคือนักกฎหมายที่จะตอบคำถามเกี่ยวกับกฎหมาย จงตอบคำถามโดยใช้ความรู้ที่ให้ดังต่อไปนี้
ถ้าหากคุณไม่รู้คำตอบ ให้ตอบว่าไม่รู้ อย่าสร้างคำตอบขึ้นมาเอง
ความรู้ที่ให้:
พระราชบัญญัติแรงงานสัมพันธ์ (ฉบับที่ 3) พ.ศ. 2544 - หมวด 6 (สมาคมนายจ้าง)

มาตรา 82  สมาคมนายจ้างย่อมเลิกด้วยเหตุใดเหตุหนึ่ง ดังต่อไปนี้
(1) ถ้ามีข้อบังคับของสมาคมนายจ้างกำหนดให้เลิกในกรณีใด เมื่อมีกรณีนั้น
(2) เมื่อที่ประชุมใหญ่มีมติให้เลิก
(3) เมื่อนายทะเบียนมีคำสั่งให้เลิก
(4) เมื่อล้มละลาย</s><|im_start|>user
เมื่อไหร่ที่สมาคมนายจ้างจะถือว่าเลิก</s><|im_start|>assistant"""

model.eval()
with torch.no_grad():
    # encoded_input = tokenizer(text, return_tensors='pt', add_special_tokens=False, padding="max-length").to(DEVICE) # for dynamic input length
    encoded_input = tokenizer(text, return_tensors='pt', add_special_tokens=False, truncation=True, padding="max_length", max_length=4096).to(DEVICE) # max length padding
    start_time = time.time()
    generate_kwargs = dict(
        {"input_ids": encoded_input["input_ids"],"attention_mask": encoded_input["attention_mask"]},
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
    print(f"Response time: {response_time - start_time}")
    ans = tokenizer.decode(encoded_output[0][encoded_input["input_ids"].shape[1]:])
    print(ans)
    print("\n")

print("Begin ONNX export\n")

dummy_input = encoded_input["input_ids"]
dummy_mask = encoded_input["attention_mask"]

try:
    with torch.no_grad():
        inputs = {
            "input_ids": dummy_input,
            "attention_mask": dummy_mask
        }
        symbolic_names = {0: 'batch_size', 1: 'max_seq_len'}
        torch.onnx.export(
            model,
            (inputs["input_ids"], inputs["attention_mask"]),
            onnx_export_path,
            export_params=True,
            do_constant_folding=True,
            opset_version=16,
            input_names=['input_ids', 'attention_mask'],
            output_names=['output'],
            dynamic_axes={'input_ids': symbolic_names,
                        'attention_mask':symbolic_names,
                        'output': symbolic_names}
        )
    print("done w/o error :)")
except Exception as e:
    print(e)
    print("done w/ error")