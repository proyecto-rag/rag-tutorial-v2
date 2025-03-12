from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.embeddings.bedrock import BedrockEmbeddings
import sys


def get_embedding_function():
    try:
        print("Usando modelo de embeddings: all-MiniLM-L12-v2 (HuggingFace)")
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L12-v2")
        return embeddings
    except Exception as e:
        print(f"\nError al cargar el modelo de embeddings: {str(e)}")
        print("\nAsegúrate de tener todas las dependencias instaladas correctamente.")
        print("Si sigues teniendo problemas, considera usar un modelo de embeddings alternativo.")
        sys.exit(1)
