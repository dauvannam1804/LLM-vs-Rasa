# # # import time

# # # start_time = time.time()

# # # Ví dụ sử dụng
# # # prompt = """
# # # Hãy điền tên, số điện thoại, số căn cước công dân từ câu 
# # # "Tôi tên là Dung, số điện thoại 0324593069 và CCCD 129593960391" 
# # # và trả về kết quả dưới dạng 
# # # " Tên: ...
# # #   Sđt: ...
# # # CCCD: ..." 
# # # Chỉ trả về kết quả không giải thích gì thêm.
# # # """
# # # prompt = """
# # # Hãy điền tên, số điện thoại, số căn cước công dân từ câu 
# # # "Tôi tên Xuân, sdt là 0835678910 và CCCD 789012345678" 
# # # và trả về kết quả dưới dạng 
# # # " Tên: ...
# # #   Sđt: ...
# # # CCCD: ..." 
# # # Chỉ trả về kết quả không giải thích gì thêm.
# # # """

# # # prompt = """
# # # Hãy trích xuất tên, số điện thoại, số căn cước công dân từ câu "Tôi tên Xuân, CCCD 789012345678 và sdt là 0835678910" và trả về kết quả dưới dạng 
# # # " Tên: ...
# # #   Sđt: ...
# # # CCCD: ..." 
# # # Chỉ trả về kết quả không giải thích gì thêm.
# # # """

# # # input_ids = tokenizer(prompt, return_tensors="pt")

# # # output = model.generate(**input_ids, max_new_tokens=100)
# # # print("output:\n")
# # # print(output)
# # # print("decode\n")
# # # print(tokenizer.decode(output[0]))
# # from transformers import AutoTokenizer, AutoModelForCausalLM
# # import torch

# # tokenizer = AutoTokenizer.from_pretrained("gemma-1.1-2b-it")
# # model = AutoModelForCausalLM.from_pretrained("gemma-1.1-2b-it")

# # # prompt_template = \
# # # """Kiểm tra ý định của câu sau {} có nằm trong ["chào", "mua sách", "thông tin cá nhân"]. \
# # # Nếu ý đinh có nằm trong list thì trả lời "1" còn không thì trả lời "0". \
# # # (Chỉ trả lời như yêu cầu và không giải thích gì thêm)
# # # """

# # def model_output(chat):
# #     prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

# #     inputs = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")
# #     outputs = model.generate(input_ids=inputs, max_new_tokens=150)
    
# #     bot_answer = tokenizer.decode(outputs[0])
# #     start = "<start_of_turn>model"
# #     end = "<eos>"

# #     # Tìm vị trí bắt đầu của chuỗi con
# #     start_index = bot_answer.find(start) + len(start)
# #     # Tìm vị trí kết thúc của chuỗi con
# #     end_index = bot_answer.find(end)

# #     # Tách văn bản giữa hai vị trí đó
# #     result = bot_answer[start_index:end_index].strip()
# #     return result

# # intent_prompt = \
# # """Phân loại câu sau "{}" thuộc lớp nào trong ["CHÀO", "MUA SÁCH", "THÔNG TIN CÁ NHÂN"].
# # Chỉ trả lời tên lớp ở trên và không thêm gì khác vào câu trả lời \
# # ví dụ: 
# # "xin chào" thì sẽ trả lời "CHÀO" \
# # "tôi muốn mua sách thì sẽ trả lời "MUA SÁCH" \
# # "tôi tên Nam, sđt 0334529001 và địa chỉ 14/16 Pvđ" thì trả lời "THÔNG TIN CÁ NHÂN" \
# # """

# # normal_prompt = \
# # """
# # Bạn là chatbot hỗ trợ người dùng đặt mua sách ở cửa hàng sách ABC. Hãy trả lời câu sau "{}"
# # """

# # confirm_userInfo_prompt = """
# # Hãy điền tên, số điện thoại, số căn cước công dân từ câu 
# # {}
# # và trả về kết quả dưới dạng 
# # "
# # Xác nhận thông tin 
# # Tên: ...
# # Sđt: ...
# # Địa chỉ: ..." 
# # Chỉ trả về kết quả không giải thích gì thêm.
# # """


# # print("### BOT ###: Xin chào, nhà sách ABC có thể giúp gì cho bạn?")
# # while True:
# #     user_input = input("### User ###: ")
# #     if user_input == "/stop":
# #         break
    
# #     chat = [
# #         {
# #           "role": "user", 
# #           "content": intent_prompt.format(user_input)
# #         }
# #     ]
    
# #     intent = model_output(chat)

# #     print("### BOT ###:", end=" ")
# #     if intent == "CHÀO":
# #         chat = [
# #         {
# #           "role": "user", 
# #           "content": normal_prompt.format(user_input)
# #         }
# #     ]
# #         bot_answer = model_output(chat)
# #         print(bot_answer)
# #     elif intent == "MUA SÁCH":
# #         chat = [
# #         {
# #           "role": "user", 
# #           "content": normal_prompt.format(user_input)
# #         }
# #     ]
# #         bot_answer = model_output(chat)
# #         print(bot_answer)
# #     elif intent == "THÔNG TIN CÁ NHÂN":
# #         chat = [
# #         {
# #           "role": "user", 
# #           "content": confirm_userInfo_prompt.format(user_input)
# #         }
# #     ]
# #         bot_answer = model_output(chat)
# #         print(bot_answer)


# # # from sentence_transformers import SentenceTransformer
# # # model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# # # # Sentences we want to encode. Example:
# # # sentence = ['This framework generates embeddings for each input sentence']

# # # # Sentences are encoded by calling model.encode()
# # # embedding = model.encode(sentence)
# # # print(embedding)
# # # print(embedding.shape)

# # # end_time = time.time()

# # # execution_time = end_time - start_time
# # # print("Thời gian thực thi:", execution_time, "giây")


# # from langchain_huggingface import HuggingFacePipeline
# # from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# # tokenizer = AutoTokenizer.from_pretrained("gemma-1.1-2b-it")
# # model = AutoModelForCausalLM.from_pretrained("gemma-1.1-2b-it")

# # pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
# # hf = HuggingFacePipeline(pipeline=pipe)

# # import os

# # file_path = 'db\\chroma.sqlite3'
# # file_size = os.path.getsize(file_path)

# # print(f"File size: {file_size} bytes")


# # from langchain_huggingface import HuggingFacePipeline
# # from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# # tokenizer = AutoTokenizer.from_pretrained("gemma-1.1-2b-it")
# # model = AutoModelForCausalLM.from_pretrained("gemma-1.1-2b-it")



# # question = "cho tôi biết địa chỉ của công ty FPT tại Hà Nội"
# # context = """
# # Địa chỉ của FPT tại VIỆT NAM như sau:
# # 1. HANOI
# # - FPT Tower, số 10 Phố Phạm Văn Bạch, Phường Dịch Vọng, Quận Cầu Giấy, Hà Nội
# # - Điện thoại: +84 24 7300 7300
# # 2. HO CHI MINH CITY
# # - Tòa nhà FPT Tân Thuận, Lô L29B -31B - 33B, đường Tân Thuận, KCX Tân Thuận, phường Tân Thuận Đông, Quận 7, Tp. Hồ Chí Minh
# # - Điện thoại: +84 28 7300 7300
# # """
# # template = f"""
# # Bạn là chatbot hỗ trợ hỏi đáp thông tin về công ty FPT. Hãy trả lời câu hỏi "{question}" dựa trên thông tin sau.
# # "{context}"
# # """

# # def model_output(chat):
# #     prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

# #     inputs = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")
# #     outputs = model.generate(input_ids=inputs, max_new_tokens=150)
    
# #     bot_answer = tokenizer.decode(outputs[0])
# #     start = "<start_of_turn>model"
# #     end = "<eos>"

# #     # Tìm vị trí bắt đầu của chuỗi con
# #     start_index = bot_answer.find(start) + len(start)
# #     # Tìm vị trí kết thúc của chuỗi con
# #     end_index = bot_answer.find(end)

# #     # Tách văn bản giữa hai vị trí đó
# #     result = bot_answer[start_index:end_index].strip()
# #     return result

# # chat = [
# #         {
# #           "role": "user", 
# #           "content": template
# #         }
# # ]

# # print(model_output(chat))

# import glob
# from tqdm import tqdm
# import time

# import torch
# from transformers import AutoTokenizer, AutoModelForCausalLM

# from langchain.prompts import PromptTemplate
# from langchain_huggingface import HuggingFacePipeline
# from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
# from langchain.chains.retrieval_qa.base import RetrievalQA
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain.schema.document import Document
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.vectorstores import Chroma
# from langchain import hub
# from chromadb.utils import embedding_functions


# tokenizer = AutoTokenizer.from_pretrained("vinai/bartpho-word-base")
# model = AutoModelForCausalLM.from_pretrained("vinai/bartpho-word-base")

# embedding = HuggingFaceEmbeddings(
#     model_name = "vinai/bartpho-word-base"
# )