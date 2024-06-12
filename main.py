import glob
from tqdm import tqdm
import time

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain import hub
from chromadb.utils import embedding_functions

start_time = time.time()

PERSIST_DIRECTORY = "db"
EMBEDDING_MODEL = "vinai/bartpho-word-base"

# Init
# embedding = HuggingFaceEmbeddings(
#     model_name = EMBEDDING_MODEL
# )
embedding = HuggingFaceEmbeddings()

vectorstore = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embedding)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
retriever = vectorstore.as_retriever(search_kwargs={"k": 1}, search_type="similarity")

# LLM
tokenizer = AutoTokenizer.from_pretrained("gemma-1.1-7b-it")
model = AutoModelForCausalLM.from_pretrained("gemma-1.1-7b-it")
# tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
# model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=1024)
hf = HuggingFacePipeline(pipeline=pipe)

# Loads text file to database
for text_file_path in tqdm(
        glob.glob("docs/*.txt", recursive=True), desc="Processing Files", position=0
    ):
        with open(text_file_path, "r", encoding="utf-8") as text_file:
            doc = Document(
                page_content=text_file.read(), metadata={"file_path": text_file_path}
            )
            texts = text_splitter.split_documents([doc])
            vectorstore.add_documents(documents=texts)


# ask and answer
question = "Địa chỉ công ty FPT tại Hà Nội là gì?"

template = """
Bạn là chatbot hỗ trợ hỏi đáp thông tin về công ty FPT. Hãy trả lời câu hỏi "{question}" dựa trên thông tin.

"{context}"

Nếu bạn không biết câu trả lời, hãy trả lời là "Tôi không rõ câu trả lời". Không cần phải tự nghĩ ra câu trả lời.
Trả lời:
"""

qa = RetrievalQA.from_chain_type(
    llm=hf, 
    chain_type="stuff", 
    retriever=retriever, 
    chain_type_kwargs={
          "prompt": PromptTemplate(
                template=template,
                input_variables=["context", "question"],
          )}
)

# answer = qa({"query": question})
# answer = qa.invoke(question)
answer = qa.run(question)
print(answer)


end_time = time.time()

execution_time = end_time - start_time
print("Thời gian thực thi:", execution_time, "giây")