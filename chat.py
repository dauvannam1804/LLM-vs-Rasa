from langchain_huggingface import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

tokenizer = AutoTokenizer.from_pretrained("gemma-1.1-2b-it")
model = AutoModelForCausalLM.from_pretrained("gemma-1.1-2b-it")
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=1024)
hf = HuggingFacePipeline(pipeline=pipe)


question = "cho tôi biết địa chỉ của công ty FPT tại thành phố Hồ Chí Minh?"
context = """
Địa chỉ của FPT tại VIỆT NAM như sau:
1. HANOI
- FPT Tower, số 10 Phố Phạm Văn Bạch, Phường Dịch Vọng, Quận Cầu Giấy, Hà Nội
- Điện thoại: +84 24 7300 7300
2. HO CHI MINH CITY
- Tòa nhà FPT Tân Thuận, Lô L29B -31B - 33B, đường Tân Thuận, KCX Tân Thuận, phường Tân Thuận Đông, Quận 7, Tp. Hồ Chí Minh
- Điện thoại: +84 28 7300 7300
"""
template = f"""
Bạn là chatbot hỗ trợ hỏi đáp thông tin về công ty FPT. Hãy trả lời câu hỏi "{question}" dựa trên thông tin sau.
"{context}"
"""

def model_output(chat):
    prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

    inputs = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")
    outputs = model.generate(input_ids=inputs, max_new_tokens=150)
    
    bot_answer = tokenizer.decode(outputs[0])
    start = "<start_of_turn>model"
    end = "<eos>"

    # Tìm vị trí bắt đầu của chuỗi con
    start_index = bot_answer.find(start) + len(start)
    # Tìm vị trí kết thúc của chuỗi con
    end_index = bot_answer.find(end)

    # Tách văn bản giữa hai vị trí đó
    result = bot_answer[start_index:end_index].strip()
    return result

chat = [
        {
          "role": "user", 
          "content": template
        }
]

print(model_output(chat))
# print(hf(template))