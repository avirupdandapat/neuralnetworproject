"""
Query the locally hosted SmolLM2 LLM (Docker, port 12434) using LangChain.
Uses ChatOpenAI with a custom base_url for the OpenAI-compatible Docker Model Runner API.
"""

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

BASE_URL = "http://localhost:12434/engines/v1"
MODEL = "smollm2"
PROMPT = "Who is Sachin Tendulkar?"


def main():
    llm = ChatOpenAI(
        base_url=BASE_URL,
        api_key="not-needed",  # Local API doesn't require a key
        model=MODEL,
        temperature=0.7,
        max_tokens=256,
    )

    print(f"Prompt: {PROMPT}\n")
    print("Sending to LLM via LangChain...")

    messages = [HumanMessage(content=PROMPT)]
    response = llm.invoke(messages)

    print(f"Response:\n{response.content}")


if __name__ == "__main__":
    main()
