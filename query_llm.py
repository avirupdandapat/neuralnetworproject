"""
Simple script to query a locally hosted SmolLM2 LLM via Docker (port 12434).
Uses the Docker Model Runner OpenAI-compatible API.
"""

import json
import urllib.request
import urllib.error

BASE_URL = "http://localhost:12434"
PROMPT = "Who is Sachin Tendulkar?"


def query_llm(prompt: str) -> str:
    url = f"{BASE_URL}/engines/v1/chat/completions"
    body = json.dumps({
        "model": "smollm2",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 256,
    }).encode("utf-8")

    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read().decode())

    # OpenAI-style response: choices[0].message.content
    return data["choices"][0]["message"]["content"].strip()


if __name__ == "__main__":
    print(f"Prompt: {PROMPT}\n")
    print("Sending to LLM...")
    try:
        response = query_llm(PROMPT)
        print(f"Response:\n{response}")
    except urllib.error.URLError as e:
        print(f"Error: Could not reach LLM at {BASE_URL}. Is the container running?")
        print(e)
    except (KeyError, json.JSONDecodeError) as e:
        print(f"Error: Unexpected API response - {e}")