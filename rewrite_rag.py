from google_labs_html_chunker.html_chunker import HtmlChunker
from urllib.request import urlopen
import chromadb
from langchain_ollama import OllamaLLM
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

#setup huggingface model
def setup_huggingface_model():
    """Setup a non-gated HuggingFace model"""
    model_name = "microsoft/DialoGPT-medium"  # Alternative: "distilbert/distilgpt2", "gpt2"
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,  # Fixed: was torch.blflost16
        device_map="auto"
    )
    
    pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.float16
    )
    
    return pipeline

# Setup local model
def setup_local_model():
    return OllamaLLM(model="llama3.2:latest")

# ChromaDB setup
def setup_vector_store(persist_directory="./chroma_db"):
    client = chromadb.PersistentClient(path=persist_directory)
    collection_name = "current_smol_collection"
    try:
        collection = client.get_collection(name=collection_name)
        print(f"Collection: {collection_name} with {collection.count()} documents ")
        return collection, True
    except Exception:
        collection = client.create_collection(name=collection_name)
        print(f"Created collection: {collection_name}")
        return collection, False

#Loading and Chunking Website, will change to local or cloud source
def load_and_process_website_if_nededed(collection, collection_exists, url):
    if collection_exists and collection.count()>0:
        print("Using existing processed documents.")
        return
    print("Processing website contect...")
    with urlopen(url) as f:
        html = f.read().decode('utf-8')
    chunker = HtmlChunker(
        max_words_per_aggregate_passage=25,
        greedily_aggregate_sibling_nodes=True,
        html_tags_to_exclude={"noscript", "script", "style"}
    )

    passages = chunker.chunk(html)
    print(f"Storing {len(passages)} document chunks")

    try:
        collection.upsert(
            documents=passages,
            ids=[str(i) for i in range(len(passages))]
        )
    except AttributeError:
        try:
            collection.add(
                documents=passages,
                ids=[str(i) for i in range(len(passages))]
            )
        except AttributeError:
            for i, passage in enumerate(passages):
                collection.add(
                    documents=[passage],
                    ids=[str(i)]
                )
    print("Documents stored successfully")

def query_and_generate_response(collection, question, use_local = True):
    #Query Vector Store
    results = collection.query(query_texts=question, n_results=3)
    context = "\n".join([f"{i+1}. {passage}" for i, passage in enumerate(results["documents"][0])])

    #Prompt Template
    prompt_template = """You are a hyper specialised AI assistant connected to an external database. Your job is to use your reasoning capabilities and context being made available to you below to answer the User Question. If you cannot find a direct answer based on the context, tell the user and then provide the most relevant answer.
    
    User question: {}

    Context: {}

    Answer:
    """
    prompt = prompt_template.format(question, context)
    if use_local:
        local_model = setup_local_model()
        response = local_model.invoke(prompt)
        return response
    else:
        # Use HuggingFace model
        pipeline = setup_huggingface_model()
        
        # Generate response
        messages = [{"role": "user", "content": prompt}]
        
        # Fixed: Temperature -> temperature (lowercase)
        outputs = pipeline(
            prompt, 
            max_new_tokens=256, 
            do_sample=True, 
            temperature=0.1,
            pad_token_id=pipeline.tokenizer.eos_token_id
        )
        
        return outputs[0]["generated_text"][len(prompt):]
    
def main():
    print("Setting up RAG Engine...")
    #Setup persistent vector store
    collection, collection_exists = setup_vector_store()
    #Load and process website if needed
    url = "https://en.wikipedia.org/wiki/Drymaeus_poecilus"
    load_and_process_website_if_nededed(collection, collection_exists, url)
    #Query and generate response
    user_question = "Can you list the regions in Brazil in which this snail's presence has been recorded?"

    print(f"User Question: {user_question}")
    print("Generating response...")
    response = query_and_generate_response(collection, user_question)
    print("\nResponse from local model:")
    print(response)

def reset_database():
    import shutil
    persist_dir = "./chroma_db"
    if os.path.exists(persist_dir):
        shutil.rmtree(persist_dir)
        print(f"Deleted directory: {persist_dir}")
    else:
        print(f"Directory does not exist: {persist_dir}")

if __name__ == "__main__":
    import sys
    if len(sys.argv)>1 and sys.argv[1] == "--reset":
        reset_database()
    else:
        main()
