import argparse
from dataclasses import dataclass
from langchain.vectorstores.chroma import Chroma
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain.prompts import ChatPromptTemplate
from functions.get_embedding_function import get_embedding_function

#from langchain.chat_models import ChatOpenAI
from gpt4all import GPT4All


CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""


def main(query_text=False):

    model_name = "all-MiniLM-L6-v2.gguf2.f16.gguf"
    gpt4all_kwargs = {'allow_download': 'True'}


    # Prepare the DB.
    #embedding_function = GPT4AllEmbeddings(model_name = model_name,gpt4all_kwargs=gpt4all_kwargs)
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=get_embedding_function())

    # Search the DB.
    results = db.similarity_search_with_relevance_scores(query_text, k=5)
    if len(results) == 0: #or results[0][1] < 0.7:
        print(f"Unable to find matching results.")
        return

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    print(prompt)


    #model = ChatOpenAI()
    model = GPT4All(model_name='Nous-Hermes-2-Mistral-7B-DPO.Q4_0.gguf', model_path='C:\\Users\\incar\\AppData\\Local\\nomic.ai\\GPT4All')
    #response_text = model.predict(prompt)
    response_text = model.generate(prompt, temp=0)
   
    sources = [doc.metadata.get("source", None) for doc, _score in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    print(formatted_response)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text
    main(query_text)
