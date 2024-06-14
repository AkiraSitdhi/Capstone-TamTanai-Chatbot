import torch
from auto_gptq import AutoGPTQForCausalLM
from transformers import AutoTokenizer, pipeline

device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Define quantized model directory
output_path = "<output path>"

# Reload model and tokenizer
model = AutoGPTQForCausalLM.from_quantized(
    output_path,
    device=device,
    use_triton=True,
    use_safetensors=True,
)
tokenizer = AutoTokenizer.from_pretrained(output_path)

prompt = """<s><|im_start|>system
คุณคือนักกฎหมายที่จะตอบคำถามเกี่ยวกับกฎหมาย จงตอบคำถามโดยใช้ความรู้ที่ให้ดังต่อไปนี้
ถ้าหากคุณไม่รู้คำตอบ ให้ตอบว่าไม่รู้ อย่าสร้างคำตอบขึ้นมาเอง
ความรู้ที่ให้:
พระราชบัญญัติแรงงานสัมพันธ์ (ฉบับที่ 3) พ.ศ. 2544 - หมวด 6 (สมาคมนายจ้าง)

มาตรา 82  สมาคมนายจ้างย่อมเลิกด้วยเหตุใดเหตุหนึ่ง ดังต่อไปนี้
(1) ถ้ามีข้อบังคับของสมาคมนายจ้างกำหนดให้เลิกในกรณีใด เมื่อมีกรณีนั้น
(2) เมื่อที่ประชุมใหญ่มีมติให้เลิก
(3) เมื่อนายทะเบียนมีคำสั่งให้เลิก
(4) เมื่อล้มละลาย</s><|im_start|>user
เมื่อไหร่ที่สมาคมนายจ้างจะถือว่าเลิก</s><|im_start|>assistant
"""

# Inference using model.enerate API
inputs = tokenizer(prompt, return_tensors="pt").to(device)
outputs = model.generate(
    **inputs, 
    temperature=0.7, 
    top_p=0.9, 
    repetition_penalty=1.1, 
    do_sample=True, 
    max_new_tokens=128
)
outputs = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"Result using generate:\n{outputs}")

# Inference can also be done using transformers' pipeline
generator = pipeline('text-generation', model=model, tokenizer=tokenizer)
outputs = generator(
    prompt,
    temperature=0.7,
    top_p=0.9,
    repetition_penalty=1.1,
    do_sample=True,
    max_new_tokens=128
)
outputs = outputs[0]['generated_text']
print(f"Result using pipeline:\n{outputs}\n")