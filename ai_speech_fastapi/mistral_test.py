from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM


model = OllamaLLM(model="mistral")

prompt = PromptTemplate.from_template("Explain the topic of {topic} in simple terms.")

chain = prompt | model

response = chain.invoke({"topic": "quantum computing"})

print("Response:\n", response)
