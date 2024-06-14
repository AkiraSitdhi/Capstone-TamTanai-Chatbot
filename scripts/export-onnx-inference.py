import time
import torch
import onnx
import onnxruntime as ort
from transformers import AutoTokenizer
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print('import completed\n')

model_path = "<model path>"
onnx_export_path = "<onnx export path>"

tokenizer = AutoTokenizer.from_pretrained(model_path)

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
เมื่อไหร่ที่สมาคมนายจ้างจะถือว่าเลิก</s><|im_start|>assistant
"""

print("Verify ONNX model...\n")
onnx_model = onnx.load(onnx_export_path)

# will fail if given >2GB model
onnx.checker.check_model(onnx_export_path)
print("Checking a Large ONNX Model >2GB is valid")


print("Testing inference...\n")
print(f"onnxruntime: device is {ort.get_device()}")
if "CUDAExecutionProvider" in ort.get_available_providers():
        providers=['CUDAExecutionProvider']
else:
        providers=['CPUExecutionProvider']
print(f"Using {providers}")

sess = ort.InferenceSession(onnx_export_path, providers=providers)

# Prepare the input data
inputs = tokenizer(
    text,
    return_tensors='pt',
    add_special_tokens=False,
    truncation=True,
    padding="max_length",
    max_length=4096
)