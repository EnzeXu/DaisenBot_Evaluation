import os
import requests
from dotenv import load_dotenv

load_dotenv()

OPENAI_URL = os.getenv("OPENAI_URL")
OPENAI_MODEL = os.getenv("OPENAI_MODEL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


def call_gpt(chat_payload: dict) -> dict:
    """
    chat_payload example:
    {
        "messages": [
            {"role": "system", "content": "..."},
            {"role": "user", "content": "..."}
        ]
    }
    """

    headers = {
        "Authorization": OPENAI_API_KEY,
        "Content-Type": "application/json",
    }

    body = {
        "model": OPENAI_MODEL,
        "messages": chat_payload["messages"],
        "temperature": 0.0,
    }

    response = requests.post(
        OPENAI_URL,
        headers=headers,
        json=body,
        timeout=60,
    )

    response.raise_for_status()
    return response.json()

def test_call_gpt():
    chat_payload = {
        "messages": [
            {
                "role": "system",
                "content": "You are a concise and accurate technical assistant."
            },
            {
                "role": "user",
                "content": "Why are driver.Mem events split into two groups?"
            }
        ]
    }

    result = call_gpt(chat_payload)

    reply = result["choices"][0]["message"]["content"]
    print("Model reply:")
    print(reply)


if __name__ == "__main__":
    test_call_gpt()


