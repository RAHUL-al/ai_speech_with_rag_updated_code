from langchain_huggingface import HuggingFaceEmbeddings

from huggingface_hub import login
import os

token = os.getenv("HF_API_TOKEN")

# Authenticate to Hugging Face
login(token=token)

# Load embedding model (1024-dimensional)
embedding_model = HuggingFaceEmbeddings(
    model_name="embaas/sentence-transformers-e5-large-v2"
)

# Test it
embedding = embedding_model.embed_query("Hello world")
print(len(embedding))  # ✅ should be 1024
