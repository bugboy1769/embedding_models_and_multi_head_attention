from langchain_ollama import OllamaLLM

model = OllamaLLM(
    model = "llama3.2:latest"
)

print(model.invoke("What is the capital of France?"))