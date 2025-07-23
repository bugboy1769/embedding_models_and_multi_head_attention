from google_labs_html_chunker.html_chunker import HtmlChunker
from urllib.request import urlopen
import chromadb
from langchain_ollama import OllamaLLM
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

# Option 1: Using a non-gated model (recommended)
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

# Option 2: Using local Ollama model (recommended for this use case)
def setup_local_model():
    """Setup local Ollama model"""
    return OllamaLLM(model="llama3:latest")

# ChromaDB setup with persistence
def setup_vector_store(persist_directory="./chroma_db"):
    """Setup ChromaDB vector store with persistence"""
    # Create persistent client
    chroma_client = chromadb.PersistentClient(path=persist_directory)
    
    collection_name = "solotrav"
    
    # Check if collection already exists
    try:
        collection = chroma_client.get_collection(name=collection_name)
        print(f"Found existing collection '{collection_name}' with {collection.count()} documents")
        return collection, True  # True indicates collection already existed
    except Exception:
        # Collection doesn't exist, create new one
        collection = chroma_client.create_collection(name=collection_name)
        print(f"Created new collection '{collection_name}'")
        return collection, False  # False indicates new collection created

def load_and_process_website_if_needed(collection, collection_exists, url):
    """Load and process website only if collection is empty"""
    if collection_exists and collection.count() > 0:
        print("Using existing processed documents. Skipping website processing.")
        return
    
    print("Processing website content...")
    
    # Load and chunk website
    with urlopen(url) as f:
        html = f.read().decode("utf-8")

    chunker = HtmlChunker(
        max_words_per_aggregate_passage=150,
        greedily_aggregate_sibling_nodes=True,
        html_tags_to_exclude={"noscript", "script", "style"}
    )

    passages = chunker.chunk(html)
    
    # Store documents - try different methods based on ChromaDB version
    print(f"Storing {len(passages)} document chunks...")
    
    try:
        # Try upsert first (newer ChromaDB versions)
        collection.upsert(
            documents=passages, 
            ids=[str(i) for i in range(len(passages))]
        )
    except AttributeError:
        try:
            # Try add method (older versions)
            collection.add(
                documents=passages, 
                ids=[str(i) for i in range(len(passages))]
            )
        except AttributeError:
            # Alternative approach - add documents one by one
            for i, passage in enumerate(passages):
                collection.add(
                    documents=[passage],
                    ids=[str(i)]
                )
    
    print("Documents stored successfully!")

def query_and_generate_response(collection, question, use_local=True):
    """Query vector store and generate response"""
    
    # Query vector store
    results = collection.query(query_texts=question, n_results=5)
    context = "\n".join(
        [f"{i+1}. {passage}" for i, passage in enumerate(results["documents"][0])]
    )
    
    # Prompt template
    prompt_template = """Hi, please give me answer to the following question. Use the provided context below.
In case you can't find answer in the article, just respond "I could not find the answer based on the context you provided."

User question: {}

Context:
{}

Answer:"""

    prompt = prompt_template.format(question, context)
    
    if use_local:
        # Use local Ollama model
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
    """Main function to run the RAG engine"""
    print("Setting up RAG engine...")
    
    # Setup persistent vector store
    collection, collection_exists = setup_vector_store()
    
    # Load and process website only if needed
    url = "https://framedventures.com/solo-travel-in-india/"
    load_and_process_website_if_needed(collection, collection_exists, url)
    
    # Query
    user_question = "Can you tell me specifically about Varkala? Can you also tell me about homestays in India? Answer both questions."
    
    print(f"\nQuestion: {user_question}")
    print("Generating response...")
    
    # Use local model (recommended)
    try:
        response = query_and_generate_response(collection, user_question, use_local=True)
        print("\nResponse from local model:")
        print(response)
    except Exception as e:
        print(f"Local model failed: {e}")
        print("Trying HuggingFace model...")
        
        # Fallback to HuggingFace model
        try:
            response = query_and_generate_response(collection, user_question, use_local=False)
            print("\nResponse from HuggingFace model:")
            print(response)
        except Exception as e:
            print(f"HuggingFace model also failed: {e}")

def reset_database():
    """Utility function to reset the database if needed"""
    import shutil
    persist_directory = "./chroma_db"
    if os.path.exists(persist_directory):
        shutil.rmtree(persist_directory)
        print("Database reset successfully!")
    else:
        print("No database found to reset.")

if __name__ == "__main__":
    import sys
    
    # Check if user wants to reset the database
    if len(sys.argv) > 1 and sys.argv[1] == "--reset":
        reset_database()
    else:
        main()