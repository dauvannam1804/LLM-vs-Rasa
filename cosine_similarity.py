from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForMaskedLM
import torch
import torch.nn.functional as F

tokenizer = AutoTokenizer.from_pretrained("vinai/bartpho-word-base")
model = AutoModelForCausalLM.from_pretrained("vinai/bartpho-word-base")

tokenizer1 = AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2")
model1 = AutoModelForMaskedLM.from_pretrained("sentence-transformers/all-mpnet-base-v2")

# Câu cần trích xuất đặc trưng
sentence1 = "cho tôi biết địa chỉ của công ty FPT tại Hà Nội"
sentence2 = """
Địa chỉ của FPT tại VIỆT NAM như sau:
1. HANOI
- FPT Tower, số 10 Phố Phạm Văn Bạch, Phường Dịch Vọng, Quận Cầu Giấy, Hà Nội
- Điện thoại: +84 24 7300 7300
2. HO CHI MINH CITY
- Tòa nhà FPT Tân Thuận, Lô L29B -31B - 33B, đường Tân Thuận, KCX Tân Thuận, phường Tân Thuận Đông, Quận 7, Tp. Hồ Chí Minh
- Điện thoại: +84 28 7300 7300
"""

def get_embeddings(text, tokenizer, model):
    # Token hóa văn bản
    inputs = tokenizer(text, return_tensors="pt")

    inputs.pop("token_type_ids", None)
    
    # Trích xuất embedding
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    
    # Lấy hidden state cuối cùng (đặc trưng của văn bản)
    last_hidden_states = outputs.hidden_states[-1]
    print(last_hidden_states.shape)
    
    # # Trung bình các embedding của các token để có embedding của toàn bộ câu
    sentence_embedding = torch.mean(last_hidden_states, dim=1)
    print(sentence_embedding.shape)
    return sentence_embedding

def cosine_similarity(embedding1, embedding2):
    return F.cosine_similarity(embedding1, embedding2).item()


# Lấy embedding của từng câu
embedding_A = get_embeddings(sentence1, tokenizer, model)
embedding_B = get_embeddings(sentence2, tokenizer, model)

embedding_A1 = get_embeddings(sentence1, tokenizer1, model1)
embedding_B1 = get_embeddings(sentence2, tokenizer1, model1)

# Tính độ tương đồng
similarity = cosine_similarity(embedding_A, embedding_B)
similarity1 = cosine_similarity(embedding_A1, embedding_B1)


print(f"Độ tương đồng giữa câu A và câu B dùng Vin: {similarity}")
print(f"Độ tương đồng giữa câu A và câu B dùng Default: {similarity1}")