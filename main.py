import time

start_time = time.time()

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

tokenizer = AutoTokenizer.from_pretrained("gemma-1.1-2b-it")
model = AutoModelForCausalLM.from_pretrained("gemma-1.1-2b-it")

# Ví dụ sử dụng
prompt = """
Hãy điền tên, số điện thoại, số căn cước công dân từ câu 
"Tôi tên là Dung, số điện thoại 0324593069 và CCCD 129593960391" 
và trả về kết quả dưới dạng 
" Tên: ...
  Sđt: ...
CCCD: ..." 
Chỉ trả về kết quả không giải thích gì thêm.
"""
# prompt = """
# Hãy điền tên, số điện thoại, số căn cước công dân từ câu 
# "Tôi tên Xuân, sdt là 0835678910 và CCCD 789012345678" 
# và trả về kết quả dưới dạng 
# " Tên: ...
#   Sđt: ...
# CCCD: ..." 
# Chỉ trả về kết quả không giải thích gì thêm.
# """

# prompt = """
# Hãy trích xuất tên, số điện thoại, số căn cước công dân từ câu "Tôi tên Xuân, CCCD 789012345678 và sdt là 0835678910" và trả về kết quả dưới dạng 
# " Tên: ...
#   Sđt: ...
# CCCD: ..." 
# Chỉ trả về kết quả không giải thích gì thêm.
# """
input_ids = tokenizer(prompt, return_tensors="pt")
# generated_text = generate_text(prompt)
# print(f"Văn bản sinh ra: {generated_text}")
output = model.generate(**input_ids, max_new_tokens=100)
print("output:\n")
print(output)
print("decode\n")
print(tokenizer.decode(output[0]))

end_time = time.time()

execution_time = end_time - start_time
print("Thời gian thực thi:", execution_time, "giây")