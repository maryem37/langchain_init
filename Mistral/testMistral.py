import requests
import json

response = requests.post(
    "http://localhost:11434/api/generate",
    json={
        "model": "mistral",
        "prompt": "What is AI?",
        "stream": False
    }
)

print(json.dumps(response.json(), indent=2))
