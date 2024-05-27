# from langchain_community.embeddings.ollama import OllamaEmbeddings
# from langchain_community.embeddings.bedrock import BedrockEmbeddings
# from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.embeddings import GPT4AllEmbeddings

def get_embedding_function():
    
    # embeddings = BedrockEmbeddings(
    #     credentials_profile_name="default", region_name="us-east-1"
    # )
    # embeddings = OllamaEmbeddings(model="nomic-embed-text")

    #embeddings = GPT4AllEmbeddings(model_name = "all-MiniLM-L6-v2.gguf2.f16.gguf")

    model_name = "all-MiniLM-L6-v2.gguf2.f16.gguf"
    gpt4all_kwargs = {'allow_download': 'True'}

    # Prepare the DB.
    embeddings = GPT4AllEmbeddings(model_name = model_name,gpt4all_kwargs=gpt4all_kwargs)
    return embeddings
