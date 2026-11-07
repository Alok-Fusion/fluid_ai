import os

import httpx
from dotenv import load_dotenv

load_dotenv()


def check_groq() -> None:
    api_key = os.getenv("GROQ_API_KEY", "")
    model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    if not api_key:
        print("provider: groq, status: N/A, response: GROQ_API_KEY is not set")
        return

    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "Reply with the word ok"}],
        "max_tokens": 10,
    }

    try:
        with httpx.Client(timeout=15.0) as client:
            resp = client.post(url, headers=headers, json=payload)
            body = resp.text[:100].replace("\n", " ").strip()
            print(f"provider: groq, status: {resp.status_code}, response: {body}")
    except (httpx.HTTPError, RuntimeError, OSError) as exc:
        err = str(exc)[:100].replace("\n", " ").strip()
        print(f"provider: groq, status: error, response: {err}")


def check_gemini() -> None:
    api_key = os.getenv("GEMINI_API_KEY", "")
    model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    if not api_key:
        print("provider: gemini, status: N/A, response: GEMINI_API_KEY is not set")
        return

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    headers = {
        "x-goog-api-key": api_key,
        "Content-Type": "application/json",
    }
    payload = {
        "contents": [{"role": "user", "parts": [{"text": "Reply with the word ok"}]}],
        "generationConfig": {"maxOutputTokens": 10},
    }

    try:
        with httpx.Client(timeout=15.0) as client:
            resp = client.post(url, headers=headers, json=payload)
            body = resp.text[:100].replace("\n", " ").strip()
            print(f"provider: gemini, status: {resp.status_code}, response: {body}")
    except (httpx.HTTPError, RuntimeError, OSError) as exc:
        err = str(exc)[:100].replace("\n", " ").strip()
        print(f"provider: gemini, status: error, response: {err}")


if __name__ == "__main__":
    check_groq()
    check_gemini()
