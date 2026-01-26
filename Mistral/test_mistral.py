import requests
#Define the local Ollama API endpoint

OLLAMA_API_URL="http://localhost:11434/api/generate"
payload={
    "model":"mistral",
    "prompt":"what is AI?",
    "stream":False
}
#Send request to ollama
response=requests.post(OLLAMA_API_URL,json=payload)

#print response
print(response.json()["response"])