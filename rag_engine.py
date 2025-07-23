from google_labs_html_chunker.html_chunker import HtmlChunker
from urllib.request import urlopen
import chromadb
from langchain_ollama import OllamaLLM
import torch
import transformers
from transformers import AutoModelForCausalLM
from huggingface_hub import login

login()

#mad security flaw
huggingface_token = "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"
#importing llm
tokenizer = transformers.AutoTokenizer.from_pretrained("google/gemma-1.1-7b-it")
model = AutoModelForCausalLM.from_pretrained("google/gemma-1.1-7b-it", token = huggingface_token)
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    model_kwargs={
        "torch_dtype": torch.blflost16,
        "quantization_config":{"load_in_4bit":True}
    }
)

#importing llm locally
local_model = OllamaLLM(
    model = "llama3:latest"
)

#chromadbstore
chroma_client = chromadb.Client()
collection = chroma_client.create_collection(name="solotrav")

#loading,chunking and storing website
with urlopen("https://framedventures.com/solo-travel-in-india/") as f:
    html = f.read().decode("utf-8")

chunker = HtmlChunker(
    max_words_per_aggregate_passage=200,
    greedily_aggregate_sibling_nodes=True,
    html_tags_to_exclude={"noscript", "script", "style"}
)

passages = chunker.chunk(html)
collection.add(documents=passages, ids=[str(i) for i in len(passages)])

#prompt template
prompt_template = """Hi, please give me answer to the following question. Use the provided context below.
In case you can't find answer in the article, just respond "I could not find the answer based on the context you provided."

User question: {}

Context:
{}
"""

user_question = "Can you tell me about things to keep in mind when solo travelling in India?"

results = collection.query(query_texts=user_question, n_results=3)
context = "\\n".join(
    [f"{i+1}. {passage}" for i, passage in enumerate(results["documents"][0])]
)
prompt = f"{prompt_template.format(user_question, context)}"


#catching answers
messages = [
    {"role":"user", "content":prompt}
]
prompt = pipeline.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
outputs = pipeline(prompt, max_new_tokens=256, do_sample=True, Temperature=0.1)
print(outputs[0]["generated_text"][len(prompt):])