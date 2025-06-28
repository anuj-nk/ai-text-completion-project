import os
import requests
from dotenv import load_dotenv

load_dotenv()
HF_API_TOKEN = os.getenv("HF_API_TOKEN")

API_URL = "https://api-inference.huggingface.co/models/HuggingFaceH4/zephyr-7b-beta"
headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}

def get_completion(prompt):
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 150,
            "temperature": 0.7,
            "return_full_text": False
        }
    }
    response = requests.post(API_URL, headers=headers, json=payload)
    if response.status_code != 200:
        return f"Error {response.status_code}: {response.json().get('error', 'Unknown error')}"
    try:
        return response.json()[0]["generated_text"]
    except Exception:
        return "Failed to parse response."

def main():
    print("Welcome to the Hugging Face Text Completion App!")
    while True:
        prompt = input("\nEnter a prompt (or 'exit' to quit): ").strip()
        if prompt.lower() == "exit":
            break
        if not prompt:
            print("Prompt cannot be empty.")
            continue
        result = get_completion(prompt)
        print("\nAI Response:\n", result)

if __name__ == "__main__":
    main()
